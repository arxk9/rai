//
// Created by jhwangbo on 13.12.16.
//

#ifndef RAI_EXPERIENCEACQUISITOR_HPP
#define RAI_EXPERIENCEACQUISITOR_HPP

#include <Eigen/Dense>
#include <Eigen/Core>
#include <string>
#include <boost/shared_ptr.hpp>
#include <stack>

#include "rai/memory/Trajectory.hpp"
#include "rai/memory/ReplayMemorySARS.hpp"
#include "rai/memory/ReplayMemoryS.hpp"
#include "rai/tasks/common/Task.hpp"
#include "raiCommon/utils/RandomNumberGenerator.hpp"
#include "rai/noiseModel/Noise.hpp"
#include "rai/noiseModel/Noise.hpp"
#include "rai/noiseModel/Noise.hpp"
#include "rai/function/common/Policy.hpp"
#include <rai/RAI_core>
#include <rai/RAI_Tensor.hpp>

#include <omp.h>

namespace rai {
namespace ExpAcq {

template<typename Dtype, int StateDim, int ActionDim, int CommandDim>
class CommonFunc {

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef Eigen::Matrix<Dtype, StateDim, 1> State;
  typedef Eigen::Matrix<Dtype, ActionDim, 1> Action;
  typedef Eigen::Matrix<Dtype, StateDim, -1> StateBatch;
  typedef Eigen::Matrix<Dtype, ActionDim, -1> ActionBatch;
  typedef Eigen::Matrix<Dtype, 1, -1> ValueBatch;
  typedef Eigen::Matrix<Dtype, -1, -1> InnerState;

  typedef Eigen::Matrix<Dtype, CommandDim, 1> Command;
  typedef Eigen::Matrix<Dtype, Eigen::Dynamic, 1> VectorXD;
  typedef Eigen::Matrix<Dtype, 1, Eigen::Dynamic> RowVectorXD;
  typedef Eigen::Matrix<Dtype, 2, 1> Result;
  typedef rai::Tensor<Dtype,2> Tensor2D;
  typedef rai::Tensor<Dtype,3> Tensor3D;

  using Task_ = Task::Task<Dtype, StateDim, ActionDim, CommandDim>;
  using Policy_ = FuncApprox::Policy<Dtype, StateDim, ActionDim>;
  using Trajectory_ = Memory::Trajectory<Dtype, StateDim, ActionDim>;
  using ReplayMemory_ = rai::Memory::ReplayMemorySARS<Dtype, StateDim, ActionDim>;

  template<typename NoiseType>
  static Result runEpisode(Task_ *&task,
                           Policy_ *policy,
                           NoiseType *&noise,
                           Trajectory_ &trajectory,
                           State &startingState,
                           double timeLimit,
                           ReplayMemory_ *memory = nullptr) {
    TerminationType termType = TerminationType::not_terminated;
    State state, state_tp1;
    Action action, noiseFreeAction, actionNoise;
    Result stat;
    int stepCount = 0;
    Dtype cost, costInThisEpisode = 0.0;
    state = startingState;
    noise->initializeNoise();
    task->setToParticularState(state);

    if (task->isTerminalState()) {
      termType = TerminationType::terminalState;
      trajectory.terminateTrajectoryAndUpdateValueTraj(
          TerminationType::terminalState, state, action,
          task->termValue(), task->discountFtr());
    }

    while (termType == TerminationType::not_terminated) {
      timer->startTimer("Policy evaluation");
      policy->forward(state, action);
      timer->stopTimer("Policy evaluation");
      actionNoise = noise->sampleNoise();
      action += actionNoise;
      timer->startTimer("Dynamics");
      task->takeOneStep(action, state_tp1, termType, cost);
      if (memory) memory->saveAnExperienceTuple(state, action, cost, state_tp1, termType);
      timer->stopTimer("Dynamics");
      stepCount++;
      trajectory.pushBackTrajectory(state, action, actionNoise, cost);
      costInThisEpisode += cost;

      if (task->dt() * stepCount + task->dt() * 0.5 >= timeLimit)
        termType = TerminationType::timeout;

      if (termType == TerminationType::terminalState) {
        trajectory.terminateTrajectoryAndUpdateValueTraj(
            termType, state_tp1, action,
            task->termValue(), task->discountFtr());
      } else if (termType == TerminationType::timeout) {
        trajectory.terminateTrajectoryAndUpdateValueTraj(
            termType, state_tp1, action, Dtype(0.0), task->discountFtr());
      }
      state = state_tp1;

    }
    stat[0] = Dtype(costInThisEpisode / (stepCount * task->dt()));
    stat[1] = stepCount;
    return stat;
  }

  ////////////// this method is useful when forward of a policy is expensive ////////
  template<typename NoiseType>
  static Result runEpisodeInBatch(std::vector<Task_ *> &task,
                                  Policy_ *policy,
                                  std::vector<NoiseType *> &noise,
                                  std::vector<Trajectory_> &trajectorySet,
                                  StateBatch &startingState,
                                  double timeLimit,
                                  ReplayMemory_ *memory = nullptr) {
    int numOfTraj = trajectorySet.size();
    TerminationType termType;
    noise[0]->initializeNoise();

    Result stat;
    Tensor3D states("state");
    Tensor3D actions("action");

    std::vector<TerminationType> termTypeBatch;
    State state_tp1, state_t;
    Action action_t, noiseFreeAction, actionNoise;
    double episodeTime = 0.0;
    Dtype costInThisStep, costInThisEpisode = 0.0, noiselessAverageCost;
    int stepCount = 0;
    int reducedIdx = numOfTraj;
    int idx[numOfTraj];
    bool Reduce[numOfTraj];


    termTypeBatch.resize(numOfTraj);
    states.resize(StateDim,1,numOfTraj);
    actions.resize(ActionDim,1,numOfTraj);

    states.copyDataFrom(startingState);

    for (int trajID = 0; trajID < numOfTraj; trajID++) {
      Reduce[trajID] = false;
      idx[trajID] = trajID;
      state_tp1 = startingState.col(trajID);
      if (policy->isRecurrent()) policy->reset(trajID);

      if (task[0]->isTerminalState(state_tp1)) {
        termTypeBatch[trajID] = TerminationType::terminalState;
        trajectorySet[trajID].terminateTrajectoryAndUpdateValueTraj(
            termTypeBatch[trajID], state_tp1, action_t,
            task[0]->termValue(), task[0]->discountFtr());
        Reduce[trajID] = true;
      } else {
        termTypeBatch[trajID] = TerminationType::not_terminated;
      }
    }
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    LOG(INFO) << numOfTraj;

    while (true) {
      episodeTime += task[0]->dt();
      int IDXbefore = reducedIdx;
      for (int cnt = IDXbefore - 1; cnt > -1; cnt--) {
        if (Reduce[cnt]) {
          int target = cnt;
          // remove column(target) from reduced batch
          // if recurrent, policy.kill(target)
          if (target < reducedIdx - 1) {
            //idx
            for (int i = target; i < reducedIdx - 1; i++)
              idx[i] = idx[i + 1];
          }
          states.removeBatch(target);
          policy->terminate(target);
          reducedIdx -= 1;
          Reduce[cnt] = false;
        }
      }
      if (reducedIdx == 0) /// Termination : No more job to do
        break;

      actions.resize(ActionDim,1,reducedIdx);
      timer->startTimer("Policy evaluation");
      policy->forward(states, actions);
      timer->stopTimer("Policy evaluation");

      for (int trajID = 0; trajID < reducedIdx; trajID++) {
        actionNoise = noise[0]->sampleNoise();
        state_t = states.batch(trajID);
        action_t = actions.batch(trajID) + actionNoise;
        task[0]->setToParticularState(state_t);
        termType = TerminationType::not_terminated;
        timer->startTimer("dynamics");
        task[0]->takeOneStep(action_t, state_tp1, termType, costInThisStep);
        if (memory) memory->saveAnExperienceTuple(state_t, action_t, costInThisStep, state_tp1, termType);
        timer->stopTimer("dynamics");
        stepCount++;
        costInThisEpisode += costInThisStep;
        states.batch(trajID) = state_tp1;
        termTypeBatch[idx[trajID]] = termType;
        trajectorySet[idx[trajID]].pushBackTrajectory(state_t,
                                                      action_t,
                                                      actionNoise,
                                                      costInThisStep);

        if (termTypeBatch[idx[trajID]] == TerminationType::terminalState) {
          trajectorySet[idx[trajID]].terminateTrajectoryAndUpdateValueTraj(
              termTypeBatch[idx[trajID]], state_tp1, action_t,
              task[0]->termValue(), task[0]->discountFtr());
          Reduce[trajID] = true;
        } else if (episodeTime + task[0]->dt() * 0.5 >= timeLimit) {
          termTypeBatch[idx[trajID]] = TerminationType::timeout;
          trajectorySet[idx[trajID]].terminateTrajectoryAndUpdateValueTraj(
              termTypeBatch[idx[trajID]], state_tp1, action_t,
              Dtype(0.0), task[0]->discountFtr());
          Reduce[trajID] = true;
        }
      }
    }
    if (stepCount == 0) stat[0] = 0;
    else stat[0] = costInThisEpisode / (stepCount * task[0]->dt());
    stat[1] = stepCount;
    return stat;
  }

  template<typename NoiseType>
  static Result runEpisodeInBatchParallel(std::vector<Task_ *> &taskset,
                                          Policy_ *policy,
                                          std::vector<NoiseType *> &noise,
                                          std::vector<Trajectory_> &trajectorySet,
                                          StateBatch &startingState,
                                          double timeLimit,
                                          ReplayMemory_ *memory = nullptr) {

    int numOfTraj = int(trajectorySet.size());
    TerminationType termType;

    for (auto &noise_ : noise)
      noise_->initializeNoise();

    Tensor3D states("state");
    Tensor3D actions("action");
    int h_dim;

    Result stat;
    std::vector<TerminationType> termTypeBatch;
    State state_tp;
    Action action_tp;
    int stepCount = 0;
    Dtype costInThisEpisode = 0.0, noiselessAverageCost;
    int ThreadN = int(taskset.size()); /////////////recommend it to be little bit larger than the number of CPUs

    LOG_IF(FATAL, ThreadN != noise.size())
    << "# of Noise: " << noise.size() << ", # of Thread: " << ThreadN << " mismatch";

    if (ThreadN >= numOfTraj) ThreadN = numOfTraj;

    int reducedIdx = ThreadN;
    int idx[ThreadN];
    Eigen::Array<bool, Eigen::Dynamic, 1> Occupied;
    double episodetime[ThreadN];
    Action action_t[ThreadN], action_noise[ThreadN];
    State state_t[ThreadN], state_t2[ThreadN];
    TerminationType termtype_t[ThreadN];

    Dtype cost[ThreadN];
    int trajcnt;

    termTypeBatch.resize(numOfTraj);
    states.resize(StateDim,1,ThreadN);
    actions.resize(ActionDim,1,ThreadN);
    Occupied.resize(ThreadN);

    for (int i = 0; i < ThreadN; i++) {
      Occupied(i) = false;
      idx[i] = -1;
      episodetime[i] = 0;
    }

    /////////////check initial states & initialize termTypeBatch
    for (int trajID = 0; trajID < numOfTraj; trajID++) {
      state_tp = startingState.col(trajID);

      if (taskset[0]->isTerminalState(state_tp)) {
        termTypeBatch[trajID] = TerminationType::terminalState;
        trajectorySet[trajID].terminateTrajectoryAndUpdateValueTraj(
            TerminationType::terminalState, state_tp, action_tp,
            taskset[0]->termValue(), taskset[0]->discountFtr());
      } else termTypeBatch[trajID] = TerminationType::not_terminated;
    }

    trajcnt = 0;
    bool reducing = false;
//    LOG(INFO) << numOfTraj << " , " << ThreadN;

    ///////////////////////////////////////////// run episodes
    while (true) {


      /// initialize tasks to trajectories with unterminated start state
      while (!Occupied.all() && !reducing) {
        if (termTypeBatch[trajcnt] == TerminationType::not_terminated) {
          for (int i = 0; i < ThreadN; i++) {
            if (!Occupied(i)) {
//            LOG(INFO) << "taskNo :" << i << " trajNo :" << trajcnt << " / " << numOfTraj;
              if (policy->isRecurrent()) policy->reset(i);
              Occupied(i) = true;
              state_t[i] = startingState.col(trajcnt);
              idx[i] = trajcnt;
              taskset[i]->setToParticularState(state_t[i]);
              episodetime[i] = 0;
              states.batch(i) = state_t[i];
              break;
            }
          }
        }
        trajcnt += 1;
        if (trajcnt == numOfTraj) {
          reducing = true;
          break;
        }
      }
      if (reducing) {
        int idxbefore = reducedIdx;
        for (int cnt = idxbefore - 1; cnt > -1; cnt--) {
          if (!Occupied(cnt)) {
            int target = cnt;
            // remove column(target) from reduced batch
            if (target < reducedIdx - 1) {
              //idx
              for (int i = target; i < reducedIdx - 1; i++)
                idx[i] = idx[i + 1];
              //policy
            }
            states.removeBatch(target);
            policy->terminate(target);
            reducedIdx -= 1;
            Occupied(cnt) = true;
          }
        }
      actions.resize(ActionDim,1,reducedIdx);
      }
      if (reducedIdx == 0) /// Termination : No more job to do
        break;

      timer->startTimer("Policy evaluation");
      policy->forward(states, actions);
      timer->stopTimer("Policy evaluation");

      int colID = 0;

      ///////////////////////////////Run trajectories parallel///////////////////////////////////
      for (int taskID = 0; taskID < reducedIdx; taskID++) {
        cost[taskID] = 0;
        action_noise[taskID] = noise[taskID]->sampleNoise();
        state_t[taskID] = states.batch(taskID);
        action_t[taskID] = actions.batch(taskID) + action_noise[taskID];

      }
      timer->startTimer("dynamics");
#pragma omp parallel for schedule(dynamic) reduction(+:stepCount)
      for (int taskID = 0; taskID < reducedIdx; taskID++) {
        if (termTypeBatch[idx[taskID]] != TerminationType::not_terminated)
          continue;
        termtype_t[taskID] = TerminationType::not_terminated;
        taskset[taskID]->takeOneStep(action_t[taskID], state_t2[taskID], termtype_t[taskID], cost[taskID]);
        episodetime[taskID] += taskset[taskID]->dt();
        if (memory)
          memory->saveAnExperienceTuple(state_t[taskID],
                                        action_t[taskID],
                                        cost[taskID],
                                        state_t2[taskID],
                                        termtype_t[taskID]);
        stepCount++;
        states.batch(taskID) = state_t2[taskID];

        trajectorySet[idx[taskID]].pushBackTrajectory(state_t[taskID],
                                                      action_t[taskID],
                                                      action_noise[taskID],
                                                      cost[taskID]);

        termTypeBatch[idx[taskID]] = termtype_t[taskID];
        if (termTypeBatch[idx[taskID]] == TerminationType::terminalState) {
          trajectorySet[idx[taskID]].terminateTrajectoryAndUpdateValueTraj(
              termTypeBatch[idx[taskID]], state_t2[taskID], action_t[taskID],
              taskset[taskID]->termValue(), taskset[taskID]->discountFtr());
//            LOG(INFO) << "terminate termstate:" <<taskID;
          Occupied(taskID) = false;
        } else if (episodetime[taskID] + taskset[taskID]->dt() * 0.5 >= timeLimit) {
          termTypeBatch[idx[taskID]] = TerminationType::timeout;
          trajectorySet[idx[taskID]].terminateTrajectoryAndUpdateValueTraj(
              termTypeBatch[idx[taskID]], state_t2[taskID], action_t[taskID],
              Dtype(0.0), taskset[taskID]->discountFtr());
//            LOG(INFO) << "terminate timeout:" <<taskID;
          Occupied(taskID) = false;
        }
      } ///parallel

      for (int taskID = 0; taskID < reducedIdx; taskID++) {
        costInThisEpisode += cost[taskID];
        cost[taskID] = 0;
      }
      timer->stopTimer("dynamics");
    }
    if (stepCount == 0) stat[0] = 0;
    else stat[0] = costInThisEpisode / (stepCount * taskset[0]->dt());
    stat[1] = stepCount;
    return stat;
  }

  template<typename NoiseType>
  static void takeOneStep(std::vector<Task_ *> &task,
                          Policy_ *policy,
                          std::vector<NoiseType *> &noise,
                          ReplayMemory_ *memory) {
    State state_t;
    Action action_t;
    task[0]->getState(state_t);
    policy->forward(state_t, action_t);
    action_t += noise[0]->sampleNoise();
    State state_tp1;
    Dtype cost;
    TerminationType termType_tp1;

    termType_tp1 = TerminationType::not_terminated;
    task[0]->takeOneStep(action_t, state_tp1, termType_tp1, cost);
    memory->saveAnExperienceTuple(state_t, action_t, cost, state_tp1, termType_tp1);

    if (termType_tp1 != TerminationType::not_terminated) {
      task[0]->setToInitialState();
      noise[0]->initializeNoise();
    }
  }

  template<typename NoiseType>
  static void takeOneStepInBatch(std::vector<Task_ *> &tasks,
                                 Policy_ *policy,
                                 std::vector<NoiseType *> &noises,
                                 ReplayMemory_ *memory) {
    unsigned threadN = tasks.size();

    LOG_IF(FATAL, threadN != noises.size()) << "# Noise: " << noises.size() << ", # Thread: " << threadN << " mismatch";

    Tensor3D states({StateDim, 1 , threadN}, "state");
    Tensor3D actions({ActionDim, 1 , threadN}, "action");

    State tempState;
    unsigned colId = 0;
    for (auto &task : tasks) {
      task->getState(tempState);
      states.batch(colId++) = tempState;
    }
    policy->forward(states,actions);

    State state_t[threadN], state_tp1[threadN];
    Dtype cost[threadN];
    Action action_t[threadN];
    TerminationType termType_tp1[threadN];

    for (int i = 0; i < threadN; i++) {
      termType_tp1[i] = TerminationType::not_terminated;
      action_t[i] = actions.batch(i) + noises[i]->sampleNoise();
      state_t[i] = states.batch(i);
    }

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < threadN; i++) {
      tasks[i]->takeOneStep(action_t[i], state_tp1[i], termType_tp1[i], cost[i]);
      memory->saveAnExperienceTuple(state_t[i], action_t[i], cost[i], state_tp1[i], termType_tp1[i]);
      if (termType_tp1[i] != TerminationType::not_terminated) {
        tasks[i]->setToInitialState();
        noises[i]->initializeNoise();
      }
    }
  }

};
}
}

#endif //RAI_EXPERIENCEACQUISITOR_HPP
