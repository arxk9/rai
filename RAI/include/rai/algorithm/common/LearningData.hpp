//
// Created by jhwangbo on 08/08/17.
// This class's old data is deleted when you acquire new data
//

#ifndef RAI_LearningData_HPP
#define RAI_LearningData_HPP

#include <rai/memory/Trajectory.hpp>
#include <rai/experienceAcquisitor/TrajectoryAcquisitor.hpp>
#include <rai/experienceAcquisitor/ExperienceTupleAcquisitor.hpp>
#include <rai/RAI_core>
#include <rai/noiseModel/NormalDistributionNoise.hpp>
#include <rai/function/common/StochasticPolicy.hpp>
#include <rai/common/VectorHelper.hpp>

namespace rai {
namespace Algorithm {

template<typename Dtype, int StateDim, int ActionDim>
class LearningData {

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

  using Trajectory_ = Memory::Trajectory<Dtype, StateDim, ActionDim>;
  using TrajAcquisitor_ = ExpAcq::TrajectoryAcquisitor<Dtype, StateDim, ActionDim>;
  using TupleAcquisitor_ = ExpAcq::ExperienceTupleAcquisitor<Dtype, StateDim, ActionDim>;
  using Task_ = Task::Task<Dtype, StateDim, ActionDim, 0>;
  using Noise_ = Noise::Noise<Dtype, ActionDim>;
  using Policy_ = FuncApprox::Policy<Dtype, StateDim, ActionDim>;
  using ValueFunc_ = FuncApprox::ValueFunction<Dtype, StateDim>;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  LearningData(TrajAcquisitor_ *acq) : trajAcquisitor_(acq) {
    stateTensor.setName("state");
    actionTensor.setName("sampled_oa");
    actionNoiseTensor.setName("noise_oa");
  }
  LearningData(TupleAcquisitor_ *acq) : tupleAcquisitor_(acq) {}

  void acquireVineTrajForNTimeSteps(rai::Vector<Task_ *> &task,
                                    rai::Vector<Noise_ *> &noise,
                                    Policy_ *policy,
                                    int numOfSteps,
                                    int numofjunct,
                                    int numOfBranchPerJunct,
                                    ValueFunc_ *vfunction = nullptr,
                                    int vis_lv = 0) {
    Utils::timer->startTimer("Simulation");

    rai::Vector<Trajectory_> trajectories;
    double dt = task[0]->dt();
    double timeLimit = task[0]->timeLimit();

    int numOfTra_ = std::ceil(1.1 * numOfSteps * dt / timeLimit);
    traj.resize(numOfTra_);
    StateBatch startState(StateDim, numOfTra_);
    sampleBatchOfInitial(startState, task);
    for (auto &noise : noise)
      noise->initializeNoise();
    for (auto &task : task)
      task->setToInitialState();
    for (auto &tra : traj)
      tra.clear();
    if (vis_lv > 1) task[0]->turnOnVisualization("");
    long double stepsTaken = trajAcquisitor_->stepsTaken();
    Dtype cost = trajAcquisitor_->acquire(task,
                                          policy,
                                          noise,
                                          traj,
                                          startState,
                                          timeLimit,
                                          true);
    if (vis_lv > 1) task[0]->turnOffVisualization();

    int stepsInThisLoop = int(trajAcquisitor_->stepsTaken() - stepsTaken);

    if (numOfSteps > stepsInThisLoop) {
      int stepsneeded = numOfSteps - stepsInThisLoop;
      rai::Vector<Trajectory_> tempTraj_;
      while (1) {
        int numofnewtraj = std::ceil(1.5 * stepsneeded * dt / timeLimit); // TODO: fix

        tempTraj_.resize(numofnewtraj);
        for (auto &tra : tempTraj_)
          tra.clear();

        StateBatch startState2(StateDim, numofnewtraj);
        sampleBatchOfInitial(startState2, task);

        for (auto &noise : noise)
          noise->initializeNoise();

        if (vis_lv > 1) task[0]->turnOnVisualization("");
        trajAcquisitor_->acquire(task,
                                 policy,
                                 noise,
                                 tempTraj_,
                                 startState2,
                                 timeLimit,
                                 true);
        if (vis_lv > 1) task[0]->turnOffVisualization();

        stepsInThisLoop = int(trajAcquisitor_->stepsTaken() - stepsTaken);
        stepsneeded = numOfSteps - stepsInThisLoop;
        ///merge trajectories
        traj.reserve(traj.size() + tempTraj_.size());
        traj.insert(traj.end(), tempTraj_.begin(), tempTraj_.end());

        if (stepsneeded <= 0) break;
      }
    }
    ///////////////////////////////////////VINE//////////////////////////////
    StateBatch VineStartPosition(StateDim, numofjunct);
    StateBatch rolloutstartState(StateDim, numofjunct * numOfBranchPerJunct);
    trajectories.resize(numofjunct * numOfBranchPerJunct);
    rolloutstartState.setOnes();
    rai::Vector<std::pair<int, int> > indx;
    rai::Op::VectorHelper::sampleRandomStates(traj, VineStartPosition, int(0.1 * timeLimit / dt), indx);

    for (int dataID = 0; dataID < numofjunct; dataID++)
      rolloutstartState.block(0, dataID * numOfBranchPerJunct, StateDim, numOfBranchPerJunct) =
          rolloutstartState.block(0, dataID * numOfBranchPerJunct, StateDim, numOfBranchPerJunct).array().colwise()
              * VineStartPosition.col(dataID).array();

    for (auto &tra : trajectories)
      tra.clear();
    for (auto &noise : noise)
      noise->initializeNoise();

    trajAcquisitor_->acquire(task, policy, noise, trajectories, rolloutstartState, timeLimit, true);

    ///merge trajectories into one vector
    traj.reserve(traj.size() + trajectories.size());
    traj.insert(traj.end(), trajectories.begin(), trajectories.end());

    int dataN = 0;
    for (auto &tra : traj) dataN += tra.size() - 1;

    if (policy->isRecurrent()) {

      /////Zero padding tensor//////////////////
      int maxlen = 0;
      for (auto &tra : traj)
        if (maxlen < tra.stateTraj.size() - 1) maxlen = int(tra.stateTraj.size()) - 1;

      int numtra = int(traj.size());
      stateTensor.resize(StateDim, maxlen, numtra);
      stateTensor.setZero();
      actionTensor.resize(ActionDim, maxlen, numtra);
      actionTensor.setZero();
      actionNoiseTensor.resize(ActionDim, maxlen, numtra);
      actionNoiseTensor.setZero();
      trajLength.resize(numtra);

      for (int i = 0; i < numtra; i++) {
        trajLength(i) = traj[i].stateTraj.size() - 1;
        stateTensor.partiallyFillBatch(i, traj[i].stateTraj, 1);
        actionTensor.partiallyFillBatch(i, traj[i].actionTraj, 1);
        actionNoiseTensor.partiallyFillBatch(i, traj[i].actionNoiseTraj, 1);
      }
    } else {


    }

    // TO DO: find better method for recurrent value function update
    stepsTaken = int(trajAcquisitor_->stepsTaken());
    stateBat.resize(StateDim, dataN);
    actionBat.resize(ActionDim, dataN);
    actionNoiseBat.resize(ActionDim, dataN);

    int colID = 0;
    for (int traID = 0; traID < traj.size(); traID++) {
      for (int timeID = 0; timeID < traj[traID].size() - 1; timeID++) {
        stateBat.col(colID) = traj[traID].stateTraj[timeID];
        actionBat.col(colID) = traj[traID].actionTraj[timeID];
        actionNoiseBat.col(colID++) = traj[traID].actionNoiseTraj[timeID];
      }
    }
    // update terimnal value
    if (vfunction) {
      termValueBat.resize(1, traj.size());
      termStateBat.resize(StateDim, traj.size());
      valueBat.resize(dataN);

      for (int traID = 0; traID < traj.size(); traID++)
        termStateBat.col(traID) = traj[traID].stateTraj.back();
      vfunction->forward(termStateBat, termValueBat);

      for (int traID = 0; traID < traj.size(); traID++)
        if (traj[traID].termType == TerminationType::timeout) {
          traj[traID].updateValueTrajWithNewTermValue(termValueBat(traID), task[0]->discountFtr());
        }

      int colID = 0;
      for (int traID = 0; traID < traj.size(); traID++)
        for (int timeID = 0; timeID < traj[traID].size() - 1; timeID++)
          valueBat(colID++) = traj[traID].valueTraj[timeID];
    }

    Utils::timer->stopTimer("Simulation");
  }

  void acquireTrajForNTimeSteps(rai::Vector<Task_ *> &task,
                                rai::Vector<Noise_ *> &noise,
                                Policy_ *policy,
                                int numOfSteps,
                                ValueFunc_ *vfunction = nullptr,
                                int vis_lv = 0) {
    acquireVineTrajForNTimeSteps(task, noise, policy, numOfSteps, 0, 0, vfunction, vis_lv);
  }

  long int stepsTaken() {
    long int steps = 0;
    if (trajAcquisitor_)
      steps += trajAcquisitor_->stepsTaken();
    if (tupleAcquisitor_)
      steps += tupleAcquisitor_->stepsTaken();
    return steps;
  }

  /////////////////////////// Core
  TrajAcquisitor_ *trajAcquisitor_ = nullptr;
  TupleAcquisitor_ *tupleAcquisitor_ = nullptr;
  rai::Vector<Trajectory_> traj;

  /////////////////////////// batches
  StateBatch stateBat, termStateBat;
  ValueBatch valueBat, termValueBat;
  ActionBatch actionBat, actionNoiseBat;

  /////////////////////////// Recurrent
  VectorXD trajLength;
  Tensor<Dtype, 3> stateTensor;
  Tensor<Dtype, 3> hiddenStateTensor;
  Tensor<Dtype, 3> actionTensor;
  Tensor<Dtype, 3> actionNoiseTensor;

 private:
  void sampleBatchOfInitial(StateBatch &initial, rai::Vector<Task_ *> &task) {
    for (int trajID = 0; trajID < initial.cols(); trajID++) {
      State state;
      task[0]->setToInitialState();
      task[0]->getState(state);
      initial.col(trajID) = state;
    }
  }

};

}
}

#endif //RAI_LearningData_HPP
