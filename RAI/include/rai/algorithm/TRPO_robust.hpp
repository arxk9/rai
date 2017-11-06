//
// Created by joonho on 20.03.17.
//

#ifndef RAI_TRPO_ROBUST_HPP
#define RAI_TRPO_ROBUST_HPP

#include <iostream>
#include "glog/logging.h"

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
#include <raiCommon/math/RAI_math.hpp>

// Neural network
//function approximations
#include "rai/function/common/Policy.hpp"
#include "rai/function/common/ValueFunction.hpp"
#include "rai/function/common/StochasticPolicy.hpp"
#include "rai/common/VectorHelper.hpp"

// memory
#include "rai/memory/Trajectory.hpp"

// acquisitor
#include "rai/experienceAcquisitor/TrajectoryAcquisitor_MultiThreadBatch.hpp"
#include <rai/experienceAcquisitor/TrajectoryAcquisitor_SingleThreadBatch.hpp>
#include <rai/experienceAcquisitor/TrajectoryAcquisitor_Sequential.hpp>

// common
#include "raiCommon/enumeration.hpp"
#include "raiCommon/math/inverseUsingCholesky.hpp"
#include "raiCommon/math/ConjugateGradient.hpp"
#include "math.h"
#include "rai/RAI_core"

namespace rai {
namespace Algorithm {

template<typename Dtype, int StateDim, int ActionDim>
class TRPO_gae_robust {

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
  typedef Eigen::Matrix<Dtype, -1, 1> Parameter;

  using Task_ = Task::Task<Dtype, StateDim, ActionDim, 0>;
  using Noise_ = Noise::NormalDistributionNoise<Dtype, ActionDim>;
  using Policy_ = FuncApprox::StochasticPolicy_TensorFlow<Dtype, StateDim, ActionDim>;
  using Trajectory_ = Memory::Trajectory<Dtype, StateDim, ActionDim>;
  using Acquisitor_ = ExpAcq::TrajectoryAcquisitor<Dtype, StateDim, ActionDim>;

  TRPO_gae_robust(std::vector<Task_ *> &tasks,
                  FuncApprox::ValueFunction <Dtype, StateDim> *vfunction,
                  FuncApprox::StochasticPolicy <Dtype, StateDim, ActionDim> *policy,
                  std::vector<Noise_ *> &noises,
                  Acquisitor_ *acquisitor,
                  Dtype lambda,
                  int K,
                  int numofjunctions,
                  unsigned testingTrajN,
                  Dtype Cov = 0) :
      task_(tasks),
      vfunction_(vfunction),
      policy_(policy),
      noise_(noises),
      acquisitor_(acquisitor),
      lambda_(lambda),
      testingTrajN_(testingTrajN),
      testTraj_(testingTrajN, Trajectory_()),
      numofjunct_(numofjunctions),
      K_(K),
      stepsTaken(0),
      DataN(0),
      cg_damping(0.1),
      klD_threshold(0.01), cov_in(Cov) {
    parameter_.setZero(policy_->getLPSize());
    policy_->getLP(parameter_);
    Utils::logger->addVariableToLog(2, "Nominal performance", "");
//    Utils::logger->addVariableToLog(2, "TRPOloss", "");
    termCost = task_[0]->termValue();
    discFactor = task_[0]->discountFtr();
    dt = task_[0]->dt();
    timeLimit = task_[0]->timeLimit();
    noNoiseRaw_.resize(task_.size());
    noNoise_.resize(task_.size());
    noiseBasePtr_.resize(task_.size());

    ///update input stdev

    if (cov_in != 0) {
//      stdev_o = noise_[0]->getCovariance().diagonal();
      stdev_o.setOnes();
      stdev_o *= cov_in;
      policy_->setStdev(stdev_o);
    }

    updatePolicyVar();

    for (int i = 0; i < task_.size(); i++)
      noNoise_[i] = &noNoiseRaw_[i];
  };

  ~TRPO_gae_robust() {};

  void runOneLoop(int numOfSteps) {
    iterNumber_++;
    for (int i = 0; i < task_.size(); i++)
      noiseBasePtr_[i] = noise_[i];

    //////////////// testing (not part of the algorithm) //////////////
    timer->disable();

    for (auto &tra : testTraj_)
      tra.clear();
    for (auto &noise : noiseBasePtr_)
      noise->initializeNoise();
    for (auto &task : task_)
      task->setToInitialState();

    StateBatch startState(StateDim, testingTrajN_);
    sampleBatchOfInitial(startState);

    if (vis_lv_ > 0) {
      task_[0]->turnOnVisualization("");
      if (task_[0]->shouldRecordVideo())
        task_[0]->startRecordingVideo(RAI_LOG_PATH + "/" + std::to_string(iterNumber_), "nominalPolicy");
    }
    Dtype averageCost = acquisitor_->acquire(task_,
                                             policy_,
                                             noiseBasePtr_,
                                             testTraj_,
                                             startState,
                                             timeLimit,
                                             false);
    if (vis_lv_ > 0) task_[0]->turnOffVisualization();
    if (task_[0]->shouldRecordVideo()) { task_[0]->endRecordingVideo(); }

    Utils::logger->appendData("Nominal performance",
                              float(acquisitor_->stepsTaken()),
                              float(averageCost));

    LOG(INFO) << "steps taken " << logger->getData("Nominal performance")->at(0).back()
              << ", average cost " << logger->getData("Nominal performance")->at(1).back();

    timer->enable();

    ////////////////////////////////Algorithm////////////////////////////
    LOG(INFO) << "Simulation";
    get_trajs(numOfSteps); /// run at least "numOfSteps" steps
    LOG(INFO) << "Vfunction update";
    VFupdate();
    LOG(INFO) << "Policy update";
    Dtype TRPOloss = TRPOUpdater();
  }

  void set_cg_daming(Dtype cgd) { cg_damping = cgd; }
  void set_kl_thres(Dtype thres) { klD_threshold = thres; }
  void setVisualizationLevel(int vis_lv) { vis_lv_ = vis_lv; }

 private:
  void get_trajs(int numOfSteps) {
    std::vector<Trajectory_> rollouts;
    Utils::timer->startTimer("Simulation");
    numOfTra_ = std::ceil(1.1 * numOfSteps * dt / timeLimit);
    traj_.resize(numOfTra_);
    StateBatch startState(StateDim, numOfTra_);
    sampleBatchOfInitial(startState);
    for (auto &noise : noiseBasePtr_)
      noise->initializeNoise();
    for (auto &task : task_)
      task->setToInitialState();
    for (auto &tra : traj_)
      tra.clear();
    if (vis_lv_ > 1) task_[0]->turnOnVisualization("");
//    Utils::timer->startTimer("Initial Trajectory Acquisition");
    Dtype cost = acquisitor_->acquire(task_,
                                      policy_,
                                      noiseBasePtr_,
                                      traj_,
                                      startState,
                                      timeLimit,
                                      true);
    if (vis_lv_ > 1) task_[0]->turnOffVisualization();

    int stepsInThisLoop = int(acquisitor_->stepsTaken() - stepsTaken);
//    Utils::timer->stopTimer("Initial Trajectory Acquisition");

    if (numOfSteps > stepsInThisLoop) {
      int stepsneeded = numOfSteps - stepsInThisLoop;
      std::vector<Trajectory_> tempTraj_;
      while (1) {
//        LOG(INFO) << "taking more steps :" << stepsneeded;
        int numofnewtraj = std::ceil(1.5 * stepsneeded * dt / timeLimit); // TODO: fix

        tempTraj_.resize(numofnewtraj);
        for (auto &tra : tempTraj_)
          tra.clear();

        StateBatch startState2(StateDim, numofnewtraj);
        sampleBatchOfInitial(startState2);

        for (auto &noise : noiseBasePtr_)
          noise->initializeNoise();

        if (vis_lv_ > 1) task_[0]->turnOnVisualization("");
        acquisitor_->acquire(task_,
                             policy_,
                             noiseBasePtr_,
                             tempTraj_,
                             startState2,
                             timeLimit,
                             true);
        if (vis_lv_ > 1) task_[0]->turnOffVisualization();

        stepsInThisLoop = int(acquisitor_->stepsTaken() - stepsTaken);
        stepsneeded = numOfSteps - stepsInThisLoop;
        ///merge trajectories
        traj_.reserve(traj_.size() + tempTraj_.size());
        traj_.insert(traj_.end(), tempTraj_.begin(), tempTraj_.end());

        if (stepsneeded <= 0) break;
      }
    }
    ///////////////////////////////////////VINE//////////////////////////////
    StateBatch VineStartPosition(StateDim, numofjunct_);
    StateBatch rolloutstartState(StateDim, numofjunct_ * K_);
    rollouts.resize(numofjunct_ * K_);
    rolloutstartState.setOnes();
    std::vector<std::pair<int, int> > indx;
    rai::Op::VectorHelper::sampleRandomStates(traj_, VineStartPosition, int(0.3 * timeLimit / dt), indx);

    for (int dataID = 0; dataID < numofjunct_; dataID++) {
      rolloutstartState.block(0, dataID * K_, StateDim, K_) =
          rolloutstartState.block(0, dataID * K_, StateDim, K_).array().colwise()
              * VineStartPosition.col(dataID).array();
    }

    for (auto &tra : rollouts)
      tra.clear();
    task_[0]->noisifyState(rolloutstartState);

    ///acquire K start state(With different noise)
    acquisitor_->acquire(task_, policy_, noiseBasePtr_, rollouts, rolloutstartState, dt, true);

    noise_[0]->initializeNoise();
    for (int i = 0; i < task_.size(); i++)
      noiseBasePtr_[i] = noise_[0];

    ///acquire K rollouts(With same noise)
    for (int trajID = 0; trajID < numofjunct_ * K_; trajID++)
      rolloutstartState.col(trajID) = rollouts[trajID].stateTraj.back();

    if (vis_lv_ > 1) task_[0]->turnOnVisualization("");
    Dtype loss = acquisitor_->acquire(task_, policy_, noiseBasePtr_, rollouts, rolloutstartState, timeLimit, true);
    if (vis_lv_ > 1) task_[0]->turnOffVisualization();

    ///merge trajectories into one vector
    traj_.reserve(traj_.size() + rollouts.size());
    traj_.insert(traj_.end(), rollouts.begin(), rollouts.end());
    stepsTaken = int(acquisitor_->stepsTaken());
    numOfTra_ = int(traj_.size());
    DataN = 0;
    for (auto &tra : traj_) DataN += tra.size() - 1;

    stateBat_.resize(StateDim, DataN);
    actionBat_.resize(ActionDim, DataN);
    actionNoiseBat_.resize(ActionDim, DataN);
    valueBat_.resize(DataN);
    costBat_.resize(DataN);
    termValueBat_.resize(numOfTra_);
    termValueBatOld_.resize(numOfTra_);
    termStateBat_.resize(StateDim, numOfTra_);

    int colID = 0;
    for (int traID = 0; traID < traj_.size(); traID++) {
      for (int timeID = 0; timeID < traj_[traID].size() - 1; timeID++) {
        stateBat_.col(colID) = traj_[traID].stateTraj[timeID];
        actionBat_.col(colID) = traj_[traID].actionTraj[timeID];
        costBat_(colID) = traj_[traID].costTraj[timeID];
        actionNoiseBat_.col(colID++) = traj_[traID].actionNoiseTraj[timeID];
      }
      termStateBat_.col(traID) = traj_[traID].stateTraj.back();
    }
    // update terimnal value
    vfunction_->forward(termStateBat_, termValueBat_);
    for (colID = 0; colID < traj_.size(); colID++)
      if (traj_[colID].termType == TerminationType::timeout)
        traj_[colID].updateValueTrajWithNewTermValue(termValueBat_(colID), task_[0]->discountFtr());
    Utils::timer->stopTimer("Simulation");
  }

  void VFupdate() {
    ValueBatch valuePrev(DataN), valueTest(DataN);
    Dtype loss;
    vfunction_->forward(stateBat_, valuePrev);
    mixfrac = 0.1;
    Utils::timer->startTimer("Vfunction update");
    int colID = 0;
    for (auto &tra : traj_)
      for (int timeID = 0; timeID < tra.size() - 1; timeID++)
        valueBat_(colID++) = tra.valueTraj[timeID];

    valueBat_ = valueBat_ * mixfrac + valuePrev * (1 - mixfrac);
    for (int i = 0; i < 50; i++) { // TODO : change terminal condition
      loss = vfunction_->performOneSolverIter(stateBat_, valueBat_);
    }
    Utils::timer->stopTimer("Vfunction update");
    LOG(INFO) << "value function loss : " << loss;
  }
  Dtype TRPOUpdater() {
    Utils::timer->startTimer("policy Training");
    /// Update Advantage
    advantage_.resize(DataN);
    bellmanErr_.resize(DataN);

    int dataID = 0;
    for (auto &tra : traj_) {
      ValueBatch advTra = tra.getGAE(vfunction_, discFactor, lambda_, termCost);
      advantage_.block(0, dataID, 1, advTra.cols()) = advTra;
      bellmanErr_.block(0, dataID, 1, advTra.cols()) = tra.bellmanErr;
      dataID += advTra.cols();
    }
    rai::Math::MathFunc::normalize(advantage_);

    /// Update Policy
    Parameter policy_grad = Parameter::Zero(parameter_.rows());
    Parameter Nat_grad = Parameter::Zero(parameter_.rows());
    Parameter fullstep = Parameter::Zero(parameter_.rows());

    policy_->getLP(parameter_);
    policy_->getStdev(stdev_o);

    LOG(INFO) << "stdev :" << stdev_o.transpose();
    Utils::timer->startTimer("Gradient computation");
    policy_->TRPOpg(stateBat_, actionBat_, actionNoiseBat_, advantage_, stdev_o, policy_grad);
    Utils::timer->stopTimer("Gradient computation");
    std::function<void(Eigen::Matrix<Dtype, -1, 1> &, Eigen::Matrix<Dtype, -1, 1> &)>
        fcn = std::bind(&TRPO_gae_robust::getFVP, this, std::placeholders::_1, std::placeholders::_2);

    Utils::timer->startTimer("Conjugate gradient");
    Dtype CGerror = conjugateGradient<Dtype>(fcn, policy_grad, 100, Dtype(1e-11), Nat_grad);
    Utils::timer->stopTimer("Conjugate gradient");

    LOG(INFO) << "conjugate grad error :" << CGerror;
    Dtype beta = std::sqrt(2 * klD_threshold / Nat_grad.dot(policy_grad));
    Nat_grad = -Nat_grad;

    Dtype expected = -policy_grad.dot(Nat_grad);
    fullstep = beta * Nat_grad;

    Utils::timer->startTimer("lineSearch");
    parameter_ += line_search(fullstep, expected);
    Utils::timer->stopTimer("lineSearch");

    policy_->setLP(parameter_);
    updatePolicyVar();/// save stdev & Update Noise Covariance
    Utils::timer->stopTimer("policy Training");

    return costOfParam(parameter_);
  }

  void getFVP(Eigen::Matrix<Dtype, -1, 1> &gradient, Eigen::Matrix<Dtype, -1, 1> &FVP) {
    policy_->TRPOfvp(stateBat_, actionBat_, actionNoiseBat_, advantage_, stdev_o, gradient, FVP); // TODO : test

    FVP += cg_damping * gradient;
  }

  void updatePolicyVar() {
    Action temp;
    policy_->getStdev(stdev_o);
    temp = stdev_o;
    temp = temp.array().square(); //var
    policycov = temp.asDiagonal();
    for (auto &noise : noise_)
      noise->updateCovariance(policycov);
  }

  inline VectorXD line_search(VectorXD &initialUpdate, Dtype &expected_improve) {

    int max_shrinks = 20;
    Dtype shrink_multiplier = 0.7;
    VectorXD paramUpdate = initialUpdate;
    VectorXD bestParamUpdate = initialUpdate;
    VectorXD parameterTest = parameter_ + initialUpdate;

    Dtype Initialcost = costOfParam(parameterTest);
    Dtype lowestCost = Initialcost;

    int i, best_indx = 0;
    for (i = 1; i < max_shrinks; i++) {
      paramUpdate = paramUpdate * shrink_multiplier;
      parameterTest = parameter_ + paramUpdate;

      Dtype cost = costOfParam(parameterTest);
      Dtype actual_Improve = Initialcost - cost;
      expected_improve = expected_improve * shrink_multiplier;
      Dtype ratio = actual_Improve / expected_improve;
      if (lowestCost > cost) {
        bestParamUpdate = paramUpdate;
        lowestCost = cost;
        best_indx = i;
      }
    }
    LOG(INFO) << "best_idx :" << best_indx;

    return bestParamUpdate;
  }

  void sampleBatchOfInitial(StateBatch &initial) {
    for (int trajID = 0; trajID < initial.cols(); trajID++) {
      State state;
      task_[0]->setToInitialState();
      task_[0]->getState(state);
      initial.col(trajID) = state;
    }
  }

  inline Dtype costOfParam(VectorXD &param) {
    policy_->setLP(param);
    return policy_->TRPOloss(stateBat_, actionBat_, actionNoiseBat_, advantage_, stdev_o);
  }

  /////////////////////////// Core //////////////////////////////////////////
  std::vector<Task_ *> task_;
  std::vector<Noise_ *> noise_;
  std::vector<Noise::Noise < Dtype, ActionDim> *> noiseBasePtr_;
  std::vector<Noise::Noise < Dtype, ActionDim>* > noNoise_;
  std::vector<Noise::NoNoise < Dtype, ActionDim> > noNoiseRaw_;
  FuncApprox::ValueFunction <Dtype, StateDim> *vfunction_;
  FuncApprox::StochasticPolicy <Dtype, StateDim, ActionDim> *policy_;
  Acquisitor_ *acquisitor_;
  Dtype lambda_;

  /////////////////////////// Algorithmic parameter ///////////////////
  int numOfTra_;
  int stepsTaken;
  int DataN;
  int numofjunct_;
  int K_;
  Dtype cov_in;

  Dtype mixfrac;
  Dtype klD_threshold;
  Dtype cg_damping;

  Dtype termCost;
  Dtype discFactor;
  Dtype dt;
  double timeLimit;

  /////////////////////////// batches
  StateBatch stateBat_, termStateBat_;
  ValueBatch valueBat_, termValueBat_, termValueBatOld_, costBat_;
  ActionBatch actionBat_, actionNoiseBat_;
  ValueBatch advantage_, bellmanErr_;

  /////////////////////////// trajectories //////////////////////
  std::vector<Trajectory_> testTraj_;
  std::vector<Trajectory_> traj_;
  std::vector<Trajectory_> junction_;

  /////////////////////////// Policy parameter
  VectorXD parameter_;
  Action stdev_o;
  Covariance policycov;

  /////////////////////////// plotting
  int iterNumber_ = 0;

  /////////////////////////// random number generator
  RandomNumberGenerator <Dtype> rn_;

  ///////////////////////////testing
  unsigned testingTrajN_;

  /////////////////////////// visualization
  int vis_lv_ = 0;
};

}
}
#endif //RAI_TRPO_ROBUST_HPP
