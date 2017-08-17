//
// Created by joonho on 18.03.17.
//

#ifndef RAI_AG_TREE_P_HPP
#define RAI_AG_TREE_P_HPP

#include <iostream>

#include "rai/experienceAcquisitor/ExperienceAcquisitor.hpp"
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
#include "rai/tasks/humanoidwalker/Osimtool.hpp"

// Neural network
//function approximations
#include "rai/function/common/Policy.hpp"
#include "rai/function/common/ValueFunction.hpp"

// memory
#include "rai/memory/Trajectory.hpp"

// common
#include "enumeration.hpp"
#include "math/inverseUsingCholesky.hpp"
#include "math/ConjugateGradient.hpp"
#include "rai/RAI_core"
#include "CMAES.hpp"

namespace RAI {
namespace Algorithm {

/// Temporary class used to debug Opensim simulation _ ljh

template<typename Dtype, int StateDim, int ActionDim>
class AGtreeP {

 public:

  typedef Eigen::Matrix<Dtype, StateDim, 1> State;
  typedef Eigen::Matrix<Dtype, StateDim, Eigen::Dynamic> StateBatch;
  typedef Eigen::Matrix<Dtype, ActionDim, 1> Action;
  typedef Eigen::Matrix<Dtype, ActionDim, Eigen::Dynamic> ActionBatch;
  typedef Eigen::Matrix<Dtype, StateDim + ActionDim, 1> StateAction;
  typedef Eigen::Matrix<Dtype, 1, 1> Value;
  typedef Eigen::Matrix<Dtype, 1, Eigen::Dynamic> ValueBatch;
  typedef Eigen::Matrix<Dtype, StateDim, ActionDim> JacobianStateResAct;
  typedef Eigen::Matrix<Dtype, 1, ActionDim> JacobianCostResAct;
  typedef Eigen::Matrix<Dtype, 1, StateDim + ActionDim> JacobianCostResStateAct;
  typedef Eigen::Matrix<Dtype, ActionDim, Eigen::Dynamic> JacobianActResParam;
  typedef Eigen::Matrix<Dtype, ActionDim, ActionDim> FimInActionSapce;
  typedef Eigen::Matrix<Dtype, ActionDim, ActionDim> Covariance;
  typedef Eigen::Matrix<Dtype, -1, -1> MatrixXD;
  typedef Eigen::Matrix<Dtype, -1, 1> VectorXD;

  using ExperienceAcquisitor_ = ExpAcq::ExperienceAcquisitor<Dtype, StateDim, ActionDim, 0>;
  using Task_ = Task::Task<Dtype, StateDim, ActionDim, 0>;
  using Policy_ = FuncApprox::Policy<Dtype, StateDim, ActionDim>;
  using Noise_ = Noise::Noise<Dtype, ActionDim>;
  using NoiseCovariance = Eigen::Matrix<Dtype, ActionDim, ActionDim>;
  using Trajectory_ = Memory::Trajectory<Dtype, StateDim, ActionDim>;

  AGtreeP(std::vector<Task_ *> *taskset,
          FuncApprox::ValueFunction<Dtype, StateDim> *vfunction,
          FuncApprox::Policy<Dtype, StateDim, ActionDim> *policy,
          Noise::NormalDistributionNoise<Dtype, ActionDim> *noise,
          int numOfInitialTra,
          int numOfBranches,
          int noiseDepth,
          double initialTrajTailTime,
          bool isopensim) :
      taskset_(taskset),
      vfunction_(vfunction),
      policy_(policy),
      numOfBranches_(numOfBranches),
      noise_(noise),
      numOfInitialTra_(numOfInitialTra),
      initialTraj_(numOfInitialTra),
      junctionTraj_(numOfBranches),
      testTrajectory_(10),
      branchTraj_(noiseDepth, std::vector<Trajectory_>(numOfBranches)),
      noiseDepth_(noiseDepth),
      initialTrajTailTime_(initialTrajTailTime),
      Osimenv_(isopensim) {
    parameter_.setZero(policy_->getAPSize());
    policy_->getAP(parameter_);
    jaco_.resize(ActionDim, policy_->getAPSize());
    fimCholesky_.resize(ActionDim, policy_->getAPSize());
    Utils::logger->addVariableToLog(2, "Nominal performance", "");
    noise2 = new Noise::NormalDistributionNoise<Dtype, ActionDim>(NoiseCovariance::Identity() * Dtype(1));
  }

  ~AGtreeP() {}

  void runOneLoop() {
    interationNumber_++;
    policy_->getAP(parameter_);
    Dtype dicFtr = taskset_->at(0)->discountFtr();
    Dtype timeLimit = taskset_->at(0)->timeLimit();
    Dtype dt = taskset_->at(0)->dt();
    Dtype termvalue = taskset_->at(0)->termValue();

    /// clearout trajectories
    testTraj_.clear();
    for (auto &tra : testTrajectory_) tra.clear();
    for (auto &tra : initialTraj_) tra.clear();
    for (auto &tra : junctionTraj_) tra.clear();
    for (auto &set : branchTraj_)
      for (auto &tra : set) tra.clear();

    ///////////////////////// testing (not part of the algorithm) /////////////////////////
    timer->disable();
    StateBatch startStateTest(StateDim, 10);
    Dtype Cost;

    if (Osimenv_) {
      taskset_->at(0)->setToInitialState();
      Cost = acquisitor_.runEpisode(taskset_->at(0), policy_, &noNoise_, &testTraj_, timeLimit, false);
    } /// Opensim needs custom function
    else {
      sampleRandomBathOfInitial(startStateTest);
      Cost = acquisitor_.runEpisodeInBatchParallel(taskset_,
                                                   policy_,
                                                   &noNoise_,
                                                   &testTrajectory_,
                                                   startStateTest,
                                                   timeLimit,
                                                   false);
    }

    logger->appendData("Nominal performance", float(acquisitor_.stepsTaken()), float(Cost));
    LOG(INFO) << "steps taken " << logger->getData("Nominal performance")->at(0).back()
              << ", average cost " << logger->getData("Nominal performance")->at(1).back();
    timer->enable();
    ///////////////////////// stage 1: simulation //////////////////
    Utils::timer->startTimer("simulation");
    std::vector<std::vector<Dtype> > valueJunction(noiseDepth_ + 1, std::vector<Dtype>(numOfBranches_));
    std::vector<State> advTuple_state;
    std::vector<Dtype> advTuple_advantage;
    std::vector<Dtype> advTuple_importance;
    std::vector<Action> advTuple_actionNoise;
    std::vector<Action> advTuple_gradient;
    /// run initial Trajectories
    StateBatch startStateOrg(StateDim, numOfInitialTra_);

    if (Osimenv_) SetRndinitialBatch(startStateOrg); /// Opensim needs custom function
    else sampleRandomBathOfInitial(startStateOrg);
    acquisitor_.runEpisodeInBatchParallel(taskset_,
                                          policy_,
                                          &noNoise_,
                                          &initialTraj_,
                                          startStateOrg,
                                          timeLimit,
                                          true);
    LOG(INFO) << "initial trajectories are computed";

    /// update terminal value and value trajectory of the initial trajectories
    ValueBatch terminalValueOrg(1, numOfInitialTra_), terminalValueBra(1, numOfBranches_);
    StateBatch terminalStateOrg(StateDim, numOfInitialTra_), terminalStateBra(StateDim, numOfBranches_);
    RAI::Op::VectorHelper::collectTerminalStates(initialTraj_, terminalStateOrg);
    vfunction_->forward(terminalStateOrg, terminalValueOrg);

    for (int trajID = 0; trajID < numOfInitialTra_; trajID++)
      if (initialTraj_[trajID].termType == TerminationType::timeout)
        initialTraj_[trajID].propagateTailCost(terminalValueOrg(trajID), dicFtr);

    /// sample random starting points along initial trajectories and run episodes
    LOG(INFO) << "Junction";
    StateBatch startStateJunct(StateDim, numOfBranches_);
    std::vector<std::pair<int, int> > indx;
    RAI::Op::VectorHelper::sampleRandomStates(initialTraj_, startStateJunct, int(initialTrajTailTime_ / dt), indx);

    acquisitor_.runEpisodeInBatchParallel(taskset_,
                                          policy_,
                                          noise_,
                                          &junctionTraj_,
                                          startStateJunct,
                                          dt * noiseDepth_,
                                          true);

    for (int trajID = 0; trajID < numOfBranches_; trajID++)
      valueJunction[0][trajID] = initialTraj_[indx[trajID].first].valueTraj[indx[trajID].second];
    LOG(INFO) << "Branch";

    for (int depthID = 1; depthID < noiseDepth_ + 1; depthID++) {
      StateBatch nthState(StateDim, numOfBranches_);
//      LOG(INFO) << "Debug1";
      for (int trajID = 0; trajID < numOfBranches_; trajID++) {
        if (junctionTraj_[trajID].stateTraj.size() <= depthID) {
          State End = junctionTraj_[trajID].stateTraj[junctionTraj_[trajID].stateTraj.size() - 1];
          nthState.col(trajID) = End; /// this is terminal state anyway
          LOG(INFO) << "Empty";
        } else nthState.col(trajID) = junctionTraj_[trajID].stateTraj[depthID];
      }

//      LOG(INFO) << "Debug2";
      acquisitor_.runEpisodeInBatchParallel(taskset_,
                                            policy_,
                                            noise_,
                                            &branchTraj_[depthID - 1],
                                            nthState,
                                            timeLimit,
                                            true);

      LOG(INFO) << depthID << " depth Branch set done";
      RAI::Op::VectorHelper::collectTerminalStates(branchTraj_[depthID - 1], terminalStateBra);
      vfunction_->forward(terminalStateBra, terminalValueBra);

      for (int trajID = 0; trajID < numOfBranches_; trajID++) {
        if (junctionTraj_[trajID].stateTraj.size() <= depthID) continue;

        branchTraj_[depthID - 1][trajID].propagateTailCost(terminalValueBra(trajID), dicFtr);
        valueJunction[depthID][trajID] = branchTraj_[depthID - 1][trajID].valueTraj[0]; ///
        advTuple_state.push_back(junctionTraj_[trajID].stateTraj[depthID - 1]);
        advTuple_actionNoise.push_back(junctionTraj_[trajID].actionNoiseTraj[depthID - 1]);
        advTuple_advantage.push_back(valueJunction[depthID][trajID] * dicFtr ///
                                         + junctionTraj_[trajID].costTraj[depthID - 1]
                                         - valueJunction[depthID - 1][trajID]);
        advTuple_gradient.push_back(advTuple_actionNoise.back() / (advTuple_actionNoise.back().squaredNorm())
                                        * advTuple_advantage.back());
      }
    }

    RAI::Math::MathFunc::normalize(advTuple_advantage);

    //// for plotting
    stateAdvantage_.resize(StateDim, advTuple_gradient.size());
    gradAdvantage_.resize(ActionDim, advTuple_gradient.size());
    for (int i = 0; i < advTuple_gradient.size(); i++) {
      stateAdvantage_.col(i) = advTuple_state[i];
      gradAdvantage_.col(i) = -advTuple_gradient[i];
    }
    Utils::timer->stopTimer("simulation");

    ///////////////////////// stage 2: vfunction train //////////////////
    LOG(INFO) << "value function training";
    Utils::timer->startTimer("vfunction Train");

    StateBatchVtrain_.setZero(StateDim, numOfBranches_ + advTuple_gradient.size());
    valueBatchVtrain_.setZero(1, numOfBranches_ + advTuple_gradient.size());
    int colIdx = 0;
    for (int trajID = 0; trajID < numOfBranches_; trajID++)
      for (int depthID = 0; depthID < noiseDepth_ + 1; depthID++) {
        if (junctionTraj_[trajID].stateTraj.size() <= depthID) {
          continue;
        }/////////////major change
        StateBatchVtrain_.col(colIdx) = junctionTraj_[trajID].stateTraj[depthID];
        valueBatchVtrain_(colIdx++) = valueJunction[depthID][trajID];
      }

    int i = 0;
    Dtype loss0;
    while (true) {

      Dtype loss = vfunction_->performOneSolverIter(StateBatchVtrain_, valueBatchVtrain_);
      LOG_IF(INFO, i % 1000 == 0) << "value function learning loss: " << loss;
      i++;
      loss0 = loss;

      if (loss < 0.0001) {
        LOG(INFO) << "value function training done";
        break;
      }
      if (i > 300) break;
    }
    Utils::timer->stopTimer("vfunction Train");

/////////////////////////// stage 3: Policy train //////////////////
    Utils::timer->startTimer("policy Training");

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

    cholInv(noise_->getCovariance(), covInv_);
    for (auto &actionNoise: advTuple_actionNoise)
      advTuple_importance.push_back(exp(-0.5 * (actionNoise.transpose() * (covInv_ * actionNoise)).sum()));

    for (int dataID = 0; dataID < dataUse; dataID++) {
      State state = stateBatchPtrain_.col(dataID);
      JacobianCostResAct jacobianQwrtAction = -advTuple_gradient[dataID];
      /// take negative for reducing cost
      costWRTAction_.col(dataID) = jacobianQwrtAction.transpose();
      Utils::timer->startTimer("JacobianOutputWRT param");
      policy_->getJacobianAction_WRT_LP(state, jaco_);
      Utils::timer->stopTimer("JacobianOutputWRT param");
      VectorXD jacobianQwrtParam = jacobianQwrtAction * jaco_;
      Covariance noise_cov = noise_->getCovariance();
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
      Dtype learningRate = 150.0; ///
      Dtype beta = sqrt(Dtype(2) * 200.0 / naturalGradientDirection.dot(jacobianQwrtParam));
      learningRate = std::min(learningRate, beta);
      paramUpdate += learningRate * naturalGradientDirection / dataUse;
    }

    newParam = parameter_ + paramUpdate;
    policy_->setAP(newParam);
    parameter_ = newParam;

    Utils::timer->stopTimer("policy Training");
    LOG(INFO) << "Policy update done";
  }

  void SetRndinitialBatch(StateBatch &initial) {
    for (int trajID = 0; trajID < initial.cols(); trajID++) {
      State state;
      OSim::tool<Dtype, StateDim, ActionDim>::RandomInit(rn_, state);
      initial.col(trajID) = state;
    }
  }
  void sampleRandomBathOfInitial(StateBatch &initial) {
    for (int trajID = 0; trajID < initial.cols(); trajID++) {
      State state;
      taskset_->at(0)->setToInitialState();
      taskset_->at(0)->getState(state);
      initial.col(trajID) = state;

    }
  }

  ActionBatch &getGradDir() { return gradAdvantage_; }
  StateBatch &getStateBatch() { return stateAdvantage_; }
  StateBatch &getValueTrainStateBatch() { return StateBatchVtrain_; }
  ValueBatch &getValueTrainValueBatch() { return valueBatchVtrain_; }
  void setVisualizationLevel(int vis_lv) { vis_lv_ = vis_lv; }
 private:

  /////////////////////////// Core //////////////////////////////////////////
  std::vector<Task_ *> *taskset_;
  ExperienceAcquisitor_ acquisitor_;
  FuncApprox::ValueFunction<Dtype, StateDim> *vfunction_;
  FuncApprox::Policy<Dtype, StateDim, ActionDim> *policy_;

  /////////////////////////// Algorithmic parameter ///////////////////
  int numOfInitialTra_ = 1;
  int noiseDepth_ = 1;
  int numOfBranches_;
  double initialTrajTailTime_;
  bool Osimenv_ = false;

  /////////////////////////// trajectories //////////////////////
  std::vector<Trajectory_> initialTraj_, junctionTraj_;
  std::vector<std::vector<Trajectory_> > branchTraj_;
  std::vector<Trajectory_> testTrajectory_;
  Trajectory_ testTraj_;

  /////////////////////////// Noise
  Noise::NormalDistributionNoise<Dtype, ActionDim> *noise2;
  Noise::NormalDistributionNoise<Dtype, ActionDim> *noise_;
  Noise::NoNoise<Dtype, ActionDim> noNoise_;

  /////////////////////////// FIM related variables
  FimInActionSapce fimInActionSpace_, fimInActionSpaceCholesky_;
  JacobianActResParam jaco_, fimCholesky_;
  Dtype klD_threshold = 0.1;
  Covariance covInv_;

  /////////////////////////// Policy parameter
  VectorXD parameter_;

  /////////////////////////// plotting
  int interationNumber_ = 0;
  ActionBatch costWRTAction_;
  StateBatch stateBatchPtrain_;
  ActionBatch actionBatchPtrain_;
  StateBatch stateAdvantage_;
  ActionBatch gradAdvantage_;

  /////////////////////////// qfunction training
  StateBatch StateBatchVtrain_;
  ValueBatch valueBatchVtrain_;

  /////////////////////////// random number generator
  RandomNumberGenerator<Dtype> rn_; /// added <Dtype> by jh
  /////////////////////////// visualization
  int vis_lv_ = 0;

};

}
}
#endif //RAI_AG_TREE_P_HPP
