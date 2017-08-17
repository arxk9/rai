//
// Created by jhwangbo on 12.09.16.
//

#ifndef RAI_AGTREE2_HPP
#define RAI_AGTREE2_HPP

#include <iostream>

#include "rai/experienceAcquisitor/AcquisitorCommonFunc.hpp"
#include "rai/tasks/common/Task.hpp"
#include <Eigen/Core>
#include <rai/noiseModel/NormalDistributionNoise.hpp>
#include <rai/noiseModel/NoNoise.hpp>
#include <Eigen/Cholesky>
#include <Eigen/Jacobi>
#include <Eigen/Cholesky>
#include <boost/bind.hpp>
#include <math.h>
#include "rai/RAI_core"
#include <vector>
#include <stdlib.h>
#include "VectorHelper.hpp"

// Neural network
//function approximations
#include "rai/function/common/Policy.hpp"
#include "rai/function/common/ValueFunction.hpp"

// memory
#include "rai/memory/Trajectory.hpp"

// Acquisitor
#include "rai/experienceAcquisitor/TrajectoryAcquisitor.hpp"

// common
#include "enumeration.hpp"
#include "math/inverseUsingCholesky.hpp"
#include "math/ConjugateGradient.hpp"
#include "rai/RAI_core"
#include "CMAES.hpp"

namespace RAI {
namespace Algorithm {

template<typename Dtype, int StateDim, int ActionDim>
class AG_tree {

 public:

  typedef Eigen::Matrix<Dtype, StateDim, 1> State;
  typedef Eigen::Matrix<Dtype, StateDim, Eigen::Dynamic> StateBatch;
  typedef Eigen::Matrix<Dtype, ActionDim, 1> Action;
  typedef Eigen::Matrix<Dtype, ActionDim, Eigen::Dynamic> ActionBatch;
  typedef Eigen::Matrix<Dtype, 1, 1> Value;
  typedef Eigen::Matrix<Dtype, 1, Eigen::Dynamic> ValueBatch;
  typedef Eigen::Matrix<Dtype, StateDim, ActionDim> JacobianStateResAct;
  typedef Eigen::Matrix<Dtype, 1, ActionDim> JacobianCostResAct;
  typedef Eigen::Matrix<Dtype, ActionDim, Eigen::Dynamic> JacobianActResParam;
  typedef Eigen::Matrix<Dtype, ActionDim, ActionDim> FimInActionSapce;
  typedef Eigen::Matrix<Dtype, ActionDim, ActionDim> Covariance;
  typedef Eigen::Matrix<Dtype, -1, -1> MatrixXD;
  typedef Eigen::Matrix<Dtype, -1, 1> VectorXD;
  using Task_ = Task::Task<Dtype, StateDim, ActionDim, 0>;
  using Trajectory_ = Memory::Trajectory<Dtype, StateDim, ActionDim>;
  using Noise_ = Noise::NormalDistributionNoise<Dtype, ActionDim>;
  using Acquisitor_ = ExpAcq::TrajectoryAcquisitor<Dtype, StateDim, ActionDim>;

  AG_tree( std::vector<Task_*> &task,
           FuncApprox::ValueFunction<Dtype, StateDim> *vfunction,
           FuncApprox::Policy<Dtype, StateDim, ActionDim> *policy,
           std::vector<Noise_*> &noise,
           Acquisitor_* acquisitor,
           int numOfInitialTra,
           int numOfBranches,
           int noiseDepth,
           double initialTrajTailTime,
           double branchTrajTime,
           Dtype learningRate = 300.0,
           int nOfTestTraj = 1) :
      task_(task),
      vfunction_(vfunction),
      policy_(policy),
      numOfBranches_(numOfBranches),
      noise_(noise),
      acquisitor_(acquisitor),
      numOfInitialTra_(numOfInitialTra),
      initialTraj_(numOfInitialTra),
      junctionTraj_(numOfBranches),
      testTrajectory_(nOfTestTraj),
      branchTraj_(noiseDepth, std::vector<Trajectory_>(numOfBranches)),
      noiseDepth_(noiseDepth),
      initialTrajTailTime_(initialTrajTailTime),
      branchTrajTime_(branchTrajTime),
      learningRate_(learningRate),
      nOfTestTraj_(nOfTestTraj) {

    parameter_.setZero(policy_->getAPSize());
    policy_->getAP(parameter_);
    jaco_.resize(ActionDim, policy_->getAPSize());
    fimCholesky_.resize(ActionDim, policy_->getAPSize());
    Utils::logger->addVariableToLog(2, "Nominal performance", "");
    noNoiseRaw_.resize(task_.size());
    noiseBasePtr_.resize(task_.size());
    noNoise_.resize(task_.size());

    for (int i=0; i<task_.size(); i++)
      noNoise_[i] = &noNoiseRaw_[i];

    for (int i=0; i<task_.size(); i++)
      noiseBasePtr_[i] = noise_[i];
  }

  ~AG_tree() {}

  void runOneLoop() {
    interationNumber_++;
    policy_->getAP(parameter_);
    Dtype dicFtr = task_[0]->discountFtr();
    Dtype timeLimit = task_[0]->timeLimit();
    Dtype dt = task_[0]->dt();

    /// clearout trajectories
    for (auto &tra : initialTraj_) tra.clear();
    for (auto &tra : junctionTraj_) tra.clear();
    for (auto &set : branchTraj_)
      for (auto &tra : set) tra.clear();
    for (auto &tra : testTrajectory_) tra.clear();

    ///////////////////////// testing (not part of the algorithm) /////////////////////////
    timer->disable();
    StateBatch startStateTest(StateDim, nOfTestTraj_);
    sampleRandomBathOfInitial(startStateTest);
    Dtype averageCost =
        acquisitor_->acquire(task_, policy_, noNoise_, testTrajectory_, startStateTest, timeLimit, false);
    logger->appendData("Nominal performance", float(acquisitor_->stepsTaken()), float(averageCost));
    LOG(INFO) << "steps taken " << logger->getData("Nominal performance")->at(0).back()
              << ", average cost " << logger->getData("Nominal performance")->at(1).back();
    timer->enable();
    //////////////////////////////////////////////////////////////////////////////////////

    ///////////////////////// stage 1: simulation //////////////////
    Utils::timer->startTimer("simulation");
    std::vector<std::vector<Dtype> > valueJunction(noiseDepth_ + 1, std::vector<Dtype>(numOfBranches_));
    std::vector<State> advTuple_state;
    std::vector<Dtype> advTuple_advantage;
    std::vector<Dtype> advTuple_importance;
    std::vector<Dtype> advTuple_MD2;
    std::vector<Action> advTuple_actionNoise;
    std::vector<Action> advTuple_gradient;

    /// run initial Trajectories
    StateBatch startStateOrg(StateDim, numOfInitialTra_);
    sampleRandomBathOfInitial(startStateOrg);
    acquisitor_->acquire(task_, policy_, noNoise_, initialTraj_, startStateOrg, timeLimit, true);
    LOG(INFO) << "initial trajectories are computed";

    /// update terminal value and value trajectory of the initial trajectories
    ValueBatch terminalValueOrg(1, numOfInitialTra_), terminalValueBra(1, numOfBranches_);
    StateBatch terminalStateOrg(StateDim, numOfInitialTra_), terminalStateBra(StateDim, numOfBranches_);
    RAI::Op::VectorHelper::collectTerminalStates(initialTraj_, terminalStateOrg);
    vfunction_->forward(terminalStateOrg, terminalValueOrg);

    for (int trajID = 0; trajID < numOfInitialTra_; trajID++)
      if (initialTraj_[trajID].termType != TerminationType::terminalState)
        initialTraj_[trajID].updateValueTrajWithNewTermValue(terminalValueOrg(trajID), dicFtr);

    /// sample random starting points along initial trajectories and run episodes
    StateBatch startStateJunct(StateDim, numOfBranches_);
    std::vector<std::pair<int, int> > indx;
    RAI::Op::VectorHelper::sampleRandomStates(initialTraj_, startStateJunct, int(initialTrajTailTime_ / dt), indx);
    acquisitor_->acquire(task_, policy_, noiseBasePtr_, junctionTraj_, startStateJunct, dt * noiseDepth_, true);

    for (int trajID = 0; trajID < numOfBranches_; trajID++)
      valueJunction[0][trajID] = initialTraj_[indx[trajID].first].valueTraj[indx[trajID].second];


    for (int depthID = 0; depthID < noiseDepth_; depthID++) {
      StateBatch nthState(StateDim, numOfBranches_);
      for (int i = 0; i < junctionTraj_.size(); i++)
        nthState.col(i) = junctionTraj_[i].stateTraj[depthID+1];
      acquisitor_->acquire(task_, policy_, noNoise_, branchTraj_[depthID], nthState, branchTrajTime_, true);
      RAI::Op::VectorHelper::collectTerminalStates(branchTraj_[depthID], terminalStateBra);
      vfunction_->forward(terminalStateBra, terminalValueBra);

      for (int trajID = 0; trajID < numOfBranches_; trajID++) {
        branchTraj_[depthID][trajID].updateValueTrajWithNewTermValue(terminalValueBra[trajID], dicFtr);
        valueJunction[depthID+1][trajID] = branchTraj_[depthID][trajID].valueTraj[0];
        advTuple_state.push_back(junctionTraj_[trajID].stateTraj[depthID]);
        advTuple_actionNoise.push_back(junctionTraj_[trajID].actionNoiseTraj[depthID]);
        advTuple_advantage.push_back(valueJunction[depthID+1][trajID] * dicFtr
                                         + junctionTraj_[trajID].costTraj[depthID]
                                         - valueJunction[depthID][trajID]);
      }
    }


//    RAI::Math::MathFunc::normalize(advTuple_advantage);

    advTuple_gradient.resize(advTuple_advantage.size());
    unsigned advIdx = 0;
    for (int depthID = 1; depthID < noiseDepth_ + 1; depthID++) {
      for (int trajID = 0; trajID < numOfBranches_; trajID++) {
        advTuple_gradient[advIdx] = advTuple_actionNoise[advIdx] / (advTuple_actionNoise[advIdx].norm()) * advTuple_advantage[advIdx];
        advIdx++;
      }
    }
    Utils::timer->stopTimer("simulation");

    ///////////////////////// stage 2: vfunction train //////////////////
    LOG(INFO) << "value function training";
    Utils::timer->startTimer("vfunction Train");
    StateBatchVtrain_.setZero(StateDim, numOfBranches_ + advTuple_gradient.size());
    valueBatchVtrain_.setZero(1, numOfBranches_ + advTuple_gradient.size());
    int colIdx = 0;

    for (int trajID = 0; trajID < numOfBranches_; trajID++)
      for (int depthID = 0; depthID < noiseDepth_ + 1 ; depthID++) {
        StateBatchVtrain_.col(colIdx) = junctionTraj_[trajID].stateTraj[depthID];
        valueBatchVtrain_(colIdx) = valueJunction[depthID][trajID];
        colIdx++;
      }
    std::cout<<"valueBatchVtrain_"<<std::endl<<valueBatchVtrain_<<std::endl;

    for (int i = 0; i < 300; i++) {
      Dtype loss = vfunction_->performOneSolverIter(StateBatchVtrain_, valueBatchVtrain_);
      LOG_IF(INFO, i % 50 == 0) << "value function learning loss: " << loss;
      if(loss < 0.0001) break;
    }
    Utils::timer->stopTimer("vfunction Train");

    ///////////////////////// stage 3: Policy train //////////////////
    Utils::timer->startTimer("policy Training");
    Dtype terminationCost = task_[0]->termValue();
    Dtype discountFactor = task_[0]->discountFtr();

    int dataLength = advTuple_gradient.size();
    int dataUse = dataLength;
    VectorXD paramUpdate = VectorXD::Zero(policy_->getLPSize());
    VectorXD newParam = VectorXD::Zero(policy_->getLPSize());

    /// forward policy in a batch for speed
    costWRTAction_.resize(ActionDim, dataUse);
    stateBatchPtrain_.resize(StateDim, dataUse);
    actionBatchPtrain_.resize(ActionDim, dataUse);
    for (int i = 0; i < dataUse; i++)
      stateBatchPtrain_.col(i) = advTuple_state[i];
    policy_->forward(stateBatchPtrain_, actionBatchPtrain_);

    cholInv(noise_[0]->getCovariance(), covInv_);
    for(auto &actionNoise: advTuple_actionNoise) {
      advTuple_MD2.push_back((actionNoise.transpose() * (covInv_ * actionNoise)).sum());
      advTuple_importance.push_back(exp(-0.5 * advTuple_MD2.back()));
    }

    //// for plotting
    stateAdvantage_.resize(StateDim, advTuple_gradient.size());
    gradAdvantage_.resize(ActionDim, advTuple_gradient.size());
    advantageBatch_.resize(1, advTuple_gradient.size());

    for (int i = 0; i < advTuple_gradient.size(); i++) {
      stateAdvantage_.col(i) = advTuple_state[i];
      gradAdvantage_.col(i) = -advTuple_gradient[i];
      advantageBatch_(i) = advTuple_advantage[i];
    }
//    std::cout<<"gradAdvantage_ "<<std::endl<<gradAdvantage_<<std::endl;
//    std::cout<<"gradAdvantage_.max() "<<std::endl<<gradAdvantage_.maxCoeff()<<std::endl;

    for (int dataID = 0; dataID < dataUse; dataID++) {
      State state = stateBatchPtrain_.col(dataID);
      JacobianCostResAct jacobianQwrtAction = -advTuple_gradient[dataID];

      /// take negative for reducing cost
      costWRTAction_.col(dataID) = jacobianQwrtAction.transpose();
      Utils::timer->startTimer("JacobianOutputWRT param");
      policy_->getJacobianAction_WRT_LP(state, jaco_);
      Utils::timer->stopTimer("JacobianOutputWRT param");
      VectorXD jacobianQwrtParam = jacobianQwrtAction * jaco_;
      Covariance noise_cov = noise_[0]->getCovariance();
      fimInActionSpace_ = noise_cov.inverse();

      Utils::timer->startTimer("Chole and SVD");
      Eigen::LLT<FimInActionSapce> chole(fimInActionSpace_); // compute the Cholesky decomposition of A
      fimInActionSpaceCholesky_ = chole.matrixL();
      fimCholesky_ = fimInActionSpaceCholesky_.transpose() * jaco_;
      Eigen::JacobiSVD<MatrixXD> svd(fimCholesky_, Eigen::ComputeThinU | Eigen::ComputeThinV);
      MatrixXD singluarValues = svd.singularValues();
      MatrixXD vMatrix = svd.matrixV();
      FimInActionSapce
          signularValueInverseSquaredMatrix = singluarValues.cwiseInverse().array().square().matrix().asDiagonal();
      Utils::timer->stopTimer("Chole and SVD");
      VectorXD naturalGradientDirection(policy_->getLPSize());
      naturalGradientDirection =
          vMatrix * (signularValueInverseSquaredMatrix * (vMatrix.transpose() * jacobianQwrtParam));

//      paramUpdate += learningRate_ * naturalGradientDirection / dataUse;

      Dtype beta = sqrt(Dtype(2) * 0.1 / naturalGradientDirection.dot(jacobianQwrtParam));
      paramUpdate += beta * naturalGradientDirection / dataUse;

    }

    newParam = parameter_ + paramUpdate;
    policy_->setAP(newParam);
    parameter_ = newParam;
    Utils::timer->stopTimer("policy Training");
  }

  void sampleRandomBathOfInitial(StateBatch& initial){
    for (int trajID = 0; trajID < initial.cols(); trajID++) {
      State state;
      task_[0]->setToInitialState();
      task_[0]->getState(state);
      initial.col(trajID) = state;
    }
  }

  ActionBatch &getGradDir() { return gradAdvantage_; }
  StateBatch &getStateBatch() { return stateAdvantage_; }
  StateBatch &getValueTrainStateBatch() { return StateBatchVtrain_; }
  ValueBatch &getValueTrainValueBatch() { return valueBatchVtrain_; }
  std::vector<std::vector<Trajectory_> > getBranchTraj() { return branchTraj_; }
  void setVisualizationLevel(int vis_lv) { vis_lv_ = vis_lv; }

 private:

  /////////////////////////// Core //////////////////////////////////////////
  std::vector<Task_*> task_;
  FuncApprox::ValueFunction<Dtype, StateDim> *vfunction_;
  FuncApprox::Policy<Dtype, StateDim, ActionDim> *policy_;
  Acquisitor_* acquisitor_;
  std::vector<Noise_*> noise_;
  std::vector<Noise::Noise<Dtype, ActionDim>* > noNoise_;
  std::vector<Noise::NoNoise<Dtype, ActionDim> > noNoiseRaw_;
  std::vector<Noise::Noise<Dtype, ActionDim>* > noiseBasePtr_;
  Dtype learningRate_;

  /////////////////////////// Algorithmic parameter ///////////////////
  int numOfInitialTra_ = 1;
  int numOfBranches_;
  int noiseDepth_ = 1;
  double initialTrajTailTime_, branchTrajTime_;

  /////////////////////////// trajectories //////////////////////
  std::vector<Trajectory_> initialTraj_, junctionTraj_;
  std::vector<std::vector<Trajectory_> > branchTraj_;
  std::vector<Trajectory_> testTrajectory_;

  /////////////////////////// FIM related variables
  FimInActionSapce fimInActionSpace_, fimInActionSpaceCholesky_;
  JacobianActResParam jaco_, fimCholesky_;
  Dtype klD_threshold = 0.1;
  Covariance covInv_;
  int nOfTestTraj_;

  /////////////////////////// Policy parameter
  VectorXD parameter_;

  /////////////////////////// plotting
  int interationNumber_ = 0;
  ActionBatch costWRTAction_;
  StateBatch stateBatchPtrain_;
  ActionBatch actionBatchPtrain_;
  StateBatch stateAdvantage_;
  ActionBatch gradAdvantage_;
  ValueBatch advantageBatch_;

  /////////////////////////// qfunction training
  StateBatch StateBatchVtrain_;
  ValueBatch valueBatchVtrain_;

  /////////////////////////// random number generator
  RandomNumberGenerator<Dtype> rn_;

  /////////////////////////// visualization
  int vis_lv_ = 0;

};

}
}

#endif //RAI_AGSPARSE_HPP
