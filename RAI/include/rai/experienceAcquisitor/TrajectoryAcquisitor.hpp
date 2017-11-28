//
// Created by jhwangbo on 3/23/17.
//

#ifndef RAI_TRAJECTORYACQUISITOR_HPP
#define RAI_TRAJECTORYACQUISITOR_HPP
#include <rai/algorithm/common/LearningData.hpp>
#include "Acquisitor.hpp"
#include "rai/memory/Trajectory.hpp"
#include "rai/noiseModel/Noise.hpp"
#include "rai/tasks/common/Task.hpp"
#include "rai/function/common/Policy.hpp"
#include "rai/memory/ReplayMemorySARS.hpp"

namespace rai {
namespace ExpAcq {

template<typename Dtype, int StateDim, int ActionDim>
class TrajectoryAcquisitor : public Acquisitor<Dtype, StateDim, ActionDim> {

  using Task_ = Task::Task<Dtype, StateDim, ActionDim, 0>;
  using Trajectory = Memory::Trajectory<Dtype, StateDim, ActionDim>;
  using Noise_ = Noise::Noise<Dtype, ActionDim>;
  using Policy_ = FuncApprox::Policy<Dtype, StateDim, ActionDim>;
  using State = Eigen::Matrix<Dtype, StateDim, 1>;
  using Action = Eigen::Matrix<Dtype, ActionDim, 1>;
  using StateBatch = Eigen::Matrix<Dtype, StateDim, -1>;
  using ActionBatch = Eigen::Matrix<Dtype, ActionDim, -1>;
  using ReplayMemory_ = Memory::ReplayMemorySARS<Dtype, StateDim, ActionDim>;
  using ValueFunc_ = FuncApprox::ValueFunction<Dtype, StateDim>;
  using ValueBatch = Eigen::Matrix<Dtype, 1, Eigen::Dynamic>;
  using DataSet = rai::Algorithm::LearningData<Dtype, StateDim, ActionDim>;

 public:
  virtual Dtype acquire(std::vector<Task_ *> &taskset,
                        Policy_ *policy,
                        std::vector<Noise_ *> &noise,
                        std::vector<Trajectory> &trajectorySet,
                        StateBatch &startingState,
                        double timeLimit,
                        bool countStep,
                        ReplayMemory_ *memory = nullptr) = 0;

  void setData(DataSet *datain) {
    Data = datain;
  }
  void acquireNEpisodes(std::vector<Task_ *> &task,
                        std::vector<Noise_ *> &noise,
                        Policy_ *policy,
                        int numOfEpisodes,
                        ValueFunc_ *vfunction = nullptr,
                        int vis_lv = 0) {

    Utils::timer->startTimer("Simulation");
    double dt = task[0]->dt();
    double timeLimit = task[0]->timeLimit();

    traj.resize(numOfEpisodes);
    StateBatch startState(StateDim, numOfEpisodes);
    sampleBatchOfInitial(startState, task);
    for (auto &noise : noise)
      noise->initializeNoise();
    for (auto &task : task)
      task->setToInitialState();
    for (auto &tra : traj)
      tra.clear();
    if (vis_lv > 1) task[0]->turnOnVisualization("");
    long double stepsTaken = this->stepsTaken();
    Dtype cost = this->acquire(task,
                               policy,
                               noise,
                               traj,
                               startState,
                               timeLimit,
                               true);
    if (vis_lv > 1) task[0]->turnOffVisualization();

    int stepsInThisLoop = int(this->stepsTaken() - stepsTaken);
    Utils::timer->stopTimer("Simulation");
  }

  void acquireVineTrajForNTimeSteps(std::vector<Task_ *> &task,
                                    std::vector<Noise_ *> &noise,
                                    Policy_ *policy,
                                    int numOfSteps,
                                    int numofjunct,
                                    int numOfBranchPerJunct,
                                    ValueFunc_ *vfunction = nullptr,
                                    int vis_lv = 0) {

    Utils::timer->startTimer("Simulation");
    std::vector<Trajectory> trajectories;
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
    long double stepsTaken = this->stepsTaken();
    Dtype cost = this->acquire(task,
                               policy,
                               noise,
                               traj,
                               startState,
                               timeLimit,
                               true);
    if (vis_lv > 1) task[0]->turnOffVisualization();

    int stepsInThisLoop = int(this->stepsTaken() - stepsTaken);

    if (numOfSteps > stepsInThisLoop) {
      int stepsneeded = numOfSteps - stepsInThisLoop;
      std::vector<Trajectory> tempTraj_;
      while (1) {
        int numofnewtraj = std::ceil(1.5 * stepsneeded * dt / timeLimit);

        tempTraj_.resize(numofnewtraj);
        for (auto &tra : tempTraj_)
          tra.clear();

        StateBatch startState2(StateDim, numofnewtraj);
        sampleBatchOfInitial(startState2, task);

        for (auto &noise : noise)
          noise->initializeNoise();

        if (vis_lv > 1) task[0]->turnOnVisualization("");
        this->acquire(task,
                      policy,
                      noise,
                      tempTraj_,
                      startState2,
                      timeLimit,
                      true);
        if (vis_lv > 1) task[0]->turnOffVisualization();

        stepsInThisLoop = int(this->stepsTaken() - stepsTaken);
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
    std::vector<std::pair<int, int> > indx;
    rai::Op::VectorHelper::sampleRandomStates(traj, VineStartPosition, int(0.1 * timeLimit / dt), indx);

    for (int dataID = 0; dataID < numofjunct; dataID++)
      rolloutstartState.block(0, dataID * numOfBranchPerJunct, StateDim, numOfBranchPerJunct) =
          rolloutstartState.block(0, dataID * numOfBranchPerJunct, StateDim, numOfBranchPerJunct).array().colwise()
              * VineStartPosition.col(dataID).array();

    for (auto &tra : trajectories)
      tra.clear();
    for (auto &noise : noise)
      noise->initializeNoise();

    this->acquire(task, policy, noise, trajectories, rolloutstartState, timeLimit, true);

    ///merge trajectories into one vector
    traj.reserve(traj.size() + trajectories.size());
    traj.insert(traj.end(), trajectories.begin(), trajectories.end());
    Utils::timer->stopTimer("Simulation");
  }

  void acquireTrajForNTimeSteps(std::vector<Task_ *> &task,
                                std::vector<Noise_ *> &noise,
                                Policy_ *policy,
                                int numOfSteps,
                                ValueFunc_ *vfunction = nullptr,
                                int vis_lv = 0) {
    acquireVineTrajForNTimeSteps(task, noise, policy, numOfSteps, 0, 0, vfunction, vis_lv);
  }

  void computeAdvantage(Task_ *task, ValueFunc_ *vfunction, Dtype lambda, bool normalize = true) {
    int batchID = 0;
    int dataID = 0;
    if (!Data->useAdvantage) {
      Data->useAdvantage = true;
      if (Data->miniBatch) Data->miniBatch->useAdvantage = true;
    }

    Data->advantages.resize(Data->maxLen, Data->batchNum);
    Data->advantages.setZero();

    for (auto &tra : traj) {
      ValueBatch advTra = tra.getGAE(vfunction, task->discountFtr(), lambda, task->termValue());
      if (normalize) {
        rai::Math::MathFunc::normalize(advTra);
      }

      if (Data->isRecurrent) {
        //RNN
        Data->advantages.block(0, batchID, advTra.cols(), 1) = advTra.transpose();
      } else {
        //BatchN = DataN
        Data->advantages.block(0, dataID, 1, advTra.cols()) = advTra;
      }

      dataID += advTra.cols();
      batchID++;
    }
  }

  DataSet *Data = nullptr;
  std::vector<Trajectory> traj;

  void saveData(Task_ *task,
                Policy_ *policy,
                ValueFunc_ *vfunction = nullptr) {

    LOG_IF(FATAL, !Data) << "You should call setData() first";
    int dataN = 0;
    int batchN = 0;
    int maxlen = 0;

    if (policy->isRecurrent() && !Data->isRecurrent) {
      Data->isRecurrent = true;
      if (Data->miniBatch) Data->miniBatch->isRecurrent = true;
    }

    for (auto &tra : traj) dataN += tra.size() - 1;

    if (policy->isRecurrent()) {
      /////Zero padding tensor//////////////////
      for (auto &tra : traj)
        if (maxlen < tra.stateTraj.size() - 1) maxlen = int(tra.stateTraj.size()) - 1;

      batchN = int(traj.size());
      Data->resize(maxlen, batchN);
      Data->setZero();

      for (int i = 0; i < batchN; i++) {
        Data->states.partiallyFillBatch(i, traj[i].stateTraj, 1);
        Data->actions.partiallyFillBatch(i, traj[i].actionTraj, 1);
        Data->actionNoises.partiallyFillBatch(i, traj[i].actionNoiseTraj, 1);
        for (int timeID = 0; timeID < traj[i].size() - 1; timeID++) {
          Data->costs.eMat()(timeID, i) = traj[i].costTraj[timeID];
        }
        Data->lengths[i] = traj[i].stateTraj.size() - 1;
        Data->termtypes[i] = Dtype(traj[i].termType);
      }

    } else {
      maxlen = 1;
      batchN = dataN;
      Data->resize(maxlen, batchN);
      Data->setZero();

      int pos = 0;
      for (int traID = 0; traID < traj.size(); traID++) {
        for (int timeID = 0; timeID < traj[traID].size() - 1; timeID++) {
          Data->states.batch(pos) = traj[traID].stateTraj[timeID];
          Data->actions.batch(pos) = traj[traID].actionTraj[timeID];
          Data->actionNoises.batch(pos++) = traj[traID].actionNoiseTraj[timeID];
        }
      }
    }

    // update terimnal value
    if (vfunction) {
      if (!Data->useValue) {
        Data->useValue = true;
        if (Data->miniBatch) Data->miniBatch->useValue = true;
      }
      Eigen::Matrix<Dtype, 1, -1> termValueBat;
      Eigen::Matrix<Dtype, StateDim, -1> termStateBat;

      Data->values.resize(maxlen, batchN);
      termValueBat.resize(1, traj.size());
      termStateBat.resize(StateDim, traj.size());

      ///update value traj
      for (int traID = 0; traID < traj.size(); traID++)
        termStateBat.col(traID) = traj[traID].stateTraj.back();
      vfunction->forward(termStateBat, termValueBat);
      for (int traID = 0; traID < traj.size(); traID++)
        if (traj[traID].termType == TerminationType::timeout) {
          traj[traID].updateValueTrajWithNewTermValue(termValueBat(traID), task->discountFtr());
        }

      int colID = 0;
      if (policy->isRecurrent()) {
        for (int traID = 0; traID < traj.size(); traID++)
          for (int timeID = 0; timeID < traj[traID].size() - 1; timeID++)
            Data->values.eMat()(timeID, traID) = traj[traID].valueTraj[timeID];
      } else {
        for (int traID = 0; traID < traj.size(); traID++)
          for (int timeID = 0; timeID < traj[traID].size() - 1; timeID++)
            Data->values.eMat()(0, colID++) = traj[traID].valueTraj[timeID];
      }
    }
  }

 private:

  void sampleBatchOfInitial(StateBatch &initial, std::vector<Task_ *> &task) {
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

#endif //RAI_TRAJECTORYACQUISITOR_HPP
