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
  typedef Eigen::Matrix<Dtype, CommandDim, 1> Command;
  typedef Eigen::Matrix<Dtype, Eigen::Dynamic, 1> VectorXD;
  typedef Eigen::Matrix<Dtype, 1, Eigen::Dynamic> RowVectorXD;
  typedef Eigen::Matrix<Dtype, 2, 1> Result;
  typedef rai::Tensor<Dtype, 3> StateTensor;
  typedef rai::Tensor<Dtype, 3> ActionTensor;

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

  template<typename NoiseType>
  static Result runEpisodeInBatchParallel(std::vector<Task_ *> &taskset,
                                          Policy_ *policy,
                                          std::vector<NoiseType *> &noise,
                                          std::vector<Trajectory_> &trajectorySet,
                                          StateBatch &startingState,
                                          double timeLimit,
                                          ReplayMemory_ *memory = nullptr) {
    int numOfTraj = int(trajectorySet.size());
    for (auto &noise_ : noise)
      noise_->initializeNoise();
    StateTensor states("state");
    ActionTensor actions("action");

    Result stat;
    int stepCount = 0;
    Dtype costInThisEpisode = 0.0;
    int ThreadN = int(taskset.size());
    LOG_IF(FATAL, ThreadN != noise.size())
    << "# of Noise: " << noise.size() << ", # of Thread: " << ThreadN << " mismatch";
    if (ThreadN >= numOfTraj) ThreadN = numOfTraj;
    int reducedIdx = ThreadN;
    double episodetime[ThreadN];
    Action action_t[ThreadN], action_noise[ThreadN];
    State state_t[ThreadN], state_t2[ThreadN];
    TerminationType termtype_t[ThreadN];
    Dtype cost[ThreadN];
    int trajcnt = 0;
    int trajectoryID[ThreadN];
    int activeThreads = ThreadN;
    bool active[ThreadN];

    states.resize(StateDim, 1, ThreadN);
    actions.resize(ActionDim, 1, ThreadN);

    for (int i = 0; i < ThreadN; i++)
      episodetime[i] = 0;

    for (int i = 0; i < ThreadN; i++) {
      state_t[i] = startingState.col(trajcnt);
      taskset[i]->setToParticularState(state_t[i]);
      trajectoryID[i] = trajcnt++;
      active[i] = true;
    }

    while (true) {
      for (int i = 0; i < ThreadN; i++) {
        if (!active[i]) continue;

        bool isTimeout = episodetime[i] + taskset[i]->dt() * 0.5 >= timeLimit;
        bool isTerminal = taskset[i]->isTerminalState();


        while (isTerminal || isTimeout) {

          if (isTerminal) {
            trajectorySet[trajectoryID[i]].terminateTrajectoryAndUpdateValueTraj(
              TerminationType::terminalState, state_t[i], action_t[i],
              taskset[0]->termValue(), taskset[i]->discountFtr());
          } else if (isTimeout) {
            trajectorySet[trajectoryID[i]].terminateTrajectoryAndUpdateValueTraj(
              TerminationType::timeout, state_t[i], action_t[i],
              Dtype(0.0), taskset[i]->discountFtr());
          }

          if (trajcnt == numOfTraj) {
            activeThreads--;
            active[i] = false;
            break;
          } else {
            state_t[i] = startingState.col(trajcnt);
            taskset[i]->setToParticularState(state_t[i]);
            trajectoryID[i] = trajcnt++;
            episodetime[i] = 0;
            isTimeout = false;
            isTerminal = taskset[i]->isTerminalState();
          }
        }
      }

      if (activeThreads == 0) break;

      states.resize(StateDim, 1, activeThreads);
      actions.resize(ActionDim, 1, activeThreads);

      int colId = 0;
      for (int i = 0; i < ThreadN; i++)
        if (active[i]) states.batch(colId++) = state_t[i];

      timer->startTimer("Policy evaluation");
      policy->forward(states, actions);
      timer->stopTimer("Policy evaluation");

//      if(policy->isRecurrent())
//      {
//        Eigen::Matrix<Dtype,-1,1> hiddenState_t;
//        for (int taskID = 0; taskID < ThreadN; taskID++){
//          hiddenState_t = policy->getHiddenState(taskID);
//          trajectorySet[trajectoryID[taskID]].pushBackHiddenState(hiddenState_t);
//        }
//      }

      colId = 0;
      for (int taskID = 0; taskID < ThreadN; taskID++) {
        if (!active[taskID]) continue;
        cost[taskID] = 0;
        action_noise[taskID] = noise[taskID]->sampleNoise();
        action_t[taskID] = actions.batch(colId++) + action_noise[taskID];
      }

      timer->startTimer("dynamics");

#pragma omp parallel for schedule(dynamic) reduction(+:stepCount)
      for (int taskID = 0; taskID < ThreadN; taskID++) {
        if (!active[taskID]) continue;

        taskset[taskID]->takeOneStep(action_t[taskID], state_t2[taskID], termtype_t[taskID], cost[taskID]);

        if (memory)
          memory->saveAnExperienceTuple(state_t[taskID],
                                        action_t[taskID],
                                        cost[taskID],
                                        state_t2[taskID],
                                        termtype_t[taskID]);

        trajectorySet[trajectoryID[taskID]].pushBackTrajectory(state_t[taskID],
                                                               action_t[taskID],
                                                               action_noise[taskID],
                                                               cost[taskID]);
        stepCount++;

        episodetime[taskID] += taskset[taskID]->dt();
        state_t[taskID] = state_t2[taskID];
      }

      timer->stopTimer("dynamics");

      for (int taskID = 0; taskID < reducedIdx; taskID++) {
        costInThisEpisode += cost[taskID];
        cost[taskID] = 0;
      }
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

    StateTensor states({StateDim, 1, threadN}, "state");
    ActionTensor actions({ActionDim, 1, threadN}, "action");

    State tempState;
    unsigned colId = 0;
    for (auto &task : tasks) {
      task->getState(tempState);
      states.batch(colId++) = tempState;
    }
    policy->forward(states, actions);

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
