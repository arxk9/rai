//
// Created by jhwangbo on 29.03.17.
//

#ifndef RAI_DDTS_HPP
#define RAI_DDTS_HPP

#include <iostream>
#include <rai/RAI_core>
#include "rai/experienceAcquisitor/ExperienceTupleAcquisitor.hpp"
#include "rai/experienceAcquisitor/TrajectoryAcquisitor_MultiThreadBatch.hpp"
#include "rai/tasks/common/Task.hpp"
#include "rai/memory/ReplayMemorySARS.hpp"
#include "rai/memory/Trajectory.hpp"
#include <Eigen/Core>
#include <rai/noiseModel/OrnsteinUhlenbeckNoise.hpp>
#include <rai/noiseModel/NoNoise.hpp>
#include <Eigen/Cholesky>
#include <Eigen/Jacobi>
#include <Eigen/Cholesky>
#include <boost/bind.hpp>
#include <rai/experienceAcquisitor/TrajectoryAcquisitor_SingleThreadBatch.hpp>
#include <rai/experienceAcquisitor/TrajectoryAcquisitor_Sequential.hpp>

// Neural network
//function approximations
#include "rai/function/common/Policy.hpp"
#include "rai/function/common/Qfunction.hpp"

// common
#include "math/RandomNumberGenerator.hpp"
#include "enumeration.hpp"
#include "math/RAI_math.hpp"
#include "math.h"

namespace RAI {
namespace Algorithm {

template<typename Dtype, int StateDim, int ActionDim>
class DDTS {

 public:

  typedef Eigen::Matrix<Dtype, StateDim, 1> State;
  typedef Eigen::Matrix<Dtype, StateDim, Eigen::Dynamic> StateBatch;
  typedef Eigen::Matrix<Dtype, ActionDim, 1> Action;
  typedef Eigen::Matrix<Dtype, ActionDim, Eigen::Dynamic> ActionBatch;
  typedef Eigen::Matrix<Dtype, 1, Eigen::Dynamic> ValueBatch;
  typedef Eigen::Matrix<Dtype, 1, Eigen::Dynamic> ScalarBatch;
  typedef Eigen::Matrix<Dtype, -1, -1> MatrixXD;
  typedef Eigen::Matrix<Dtype, -1, 1> VectorXD;
  typedef Eigen::Matrix<Dtype, 1, -1> RowVectorXD;

  using Task_ = Task::Task<Dtype, StateDim, ActionDim, 0>;
  using ReplayMemory_ = RAI::Memory::ReplayMemorySARS<Dtype, StateDim, ActionDim>;
  using Qfunction_ = FuncApprox::Qfunction<Dtype, StateDim, ActionDim>;
  using Policy_ = FuncApprox::Policy<Dtype, StateDim, ActionDim>;
  using Noise_ = Noise::Noise<Dtype, ActionDim>;
  using Trajectory = Memory::Trajectory<Dtype, StateDim, ActionDim>;
  using Acquisitor_ = ExpAcq::TrajectoryAcquisitor<Dtype, StateDim, ActionDim>;
  using TestAcquisitor_ = ExpAcq::TrajectoryAcquisitor_Sequential<Dtype, StateDim, ActionDim>;

  DDTS(std::vector<Task_ *> &task,
       Qfunction_ *qfunction,
       Qfunction_ *qfunction_target,
       Policy_ *policy,
       Policy_ *policy_target,
       std::vector<Noise_ *> &noise,
       Acquisitor_ *acquisitor,
       ReplayMemory_ *memory,
       unsigned batchSize,
       unsigned nnUpdatePerIter,
       Dtype trajTimeLimit,
       unsigned testingTrajN = 1,
       Dtype tau = 1e-3) :
      qfunction_(qfunction),
      qfunction_target_(qfunction_target),
      policy_(policy),
      policy_target_(policy_target),
      noise_(noise),
      acquisitor_(acquisitor),
      memorySARS_(memory),
      batSize_(batchSize),
      nnUpdatePerIter_(nnUpdatePerIter),
      tau_(tau),
      testingTrajN_(testingTrajN),
      testTraj_(testingTrajN, Trajectory()),
      trajTimeLimit_(trajTimeLimit),
      task_(task) {

    for (auto &task : task_)
      task->setToInitialState();

    Utils::logger->addVariableToLog(2, "Nominal performance", "");
    qfunction_->copyAPFrom(qfunction_target_);
    parTrajSet_.resize(task_.size());
    trajectoryMemorySize_ = unsigned(10);
    preTrajId_.resize(task_.size());
    parameter_.resize(policy_->getLPSize());
  };

  ~DDTS() {};

  void runForOneIteration() {

    //////////////// testing (not part of the algorithm) ////////////////////
    Utils::timer->disable();
    for (auto &tra : testTraj_)
      tra.clear();
    for (auto &noise : noise_)
      noise->initializeNoise();
    for (auto &task : task_)
      task->setToInitialState();

    StateBatch startState(StateDim, testingTrajN_);
    sampleBatchOfInitial(startState);

    Noise::NoNoise<Dtype, ActionDim> noNoises[task_.size()];
    std::vector<Noise_ *> noiseVec;
    for (int i = 0; i < task_.size(); i++)
      noiseVec.push_back(&noNoises[i]);

    if (vis_lv_ > 0) task_[0]->turnOnVisualization("");
    Dtype averageCost = testAcquisitor_.acquire(task_,
                                                policy_,
                                                noiseVec,
                                                testTraj_,
                                                startState,
                                                task_[0]->timeLimit(),
                                                false);
    if (vis_lv_ > 0) task_[0]->turnOffVisualization();
    Utils::logger->appendData("Nominal performance",
                              float(acquisitor_->stepsTaken()),
                              float(averageCost));
    LOG(INFO) << "steps taken " << logger->getData("Nominal performance")->at(0).back()
              << ", average cost " << logger->getData("Nominal performance")->at(1).back();

    /// reset all for learning
    for (auto &task : task_)
      task->setToInitialState();
    for (auto &noise : noise_)
      noise->initializeNoise();

    Utils::timer->enable();
    /////////////////////////////////////////////////////////////////////////

    learnForOneIteration();
  }

  void setVisualizationLevel(int vis_lv) { vis_lv_ = vis_lv; }

 private:

  void sampleBatchOfInitial(StateBatch &initialBatch) {
    if (trajectoryN < trajectoryMemorySize_) {
      for (int trajID = 0; trajID < initialBatch.cols(); trajID++) {
        State state;
        task_[0]->setToInitialState();
        task_[0]->getState(state);
        initialBatch.col(trajID) = state;
      }
    } else {
      for (int i = 0; i < initialBatch.cols(); i++) {
//        unsigned trajId = rn_.intRand(0, bestTrajSet_.size() - 1);
        unsigned trajId = rn_.intWeightedRand(bestTrajSet_.size() - 1, 0.9);
        unsigned timeId;
        if (bestTrajSet_[trajId].termType == TerminationType::terminalState)
          timeId = rn_.intRand(0, bestTrajSet_[trajId].size() - 2);
        else
          timeId = rn_.intRand(0, bestTrajSet_[trajId].size() - 1);
        preTrajId_[i] = std::pair<unsigned, unsigned>(trajId, timeId);
        initialBatch.col(i) = bestTrajSet_[trajId].stateTraj[timeId];
      }
    }
  }

  void learnForOneIteration() {
    Utils::timer->startTimer("Simulation");
    for (auto& traj : parTrajSet_)
      traj.clear();
    StateBatch startState(StateDim, task_.size());
    sampleBatchOfInitial(startState);
    acquisitor_->acquire(task_,
                         policy_,
                         noise_,
                         parTrajSet_,
                         startState,
                         trajTimeLimit_,
                         true,
                         memorySARS_);
    Utils::timer->stopTimer("Simulation");

    Utils::timer->startTimer("Computing values and sorting");
    for (auto &traj : parTrajSet_)
      numOfStepsTaken_ += traj.size();

    /// prepend the old traj and update values
    for (int i = 0; i < parTrajSet_.size(); i++) {
      if (!(trajectoryN < trajectoryMemorySize_) && preTrajId_[i].second != 0) {
        parTrajSet_[i].prependAsectionOfTraj(bestTrajSet_[preTrajId_[i].first],
                                             0,
                                             preTrajId_[i].second - 1);
      }
      if (parTrajSet_[i].termType == TerminationType::terminalState)
        parTrajSet_[i].updateValueTrajWithNewTermValue(task_[0]->termValue(), task_[0]->discountFtr());
      else {
        State state = parTrajSet_[i].stateTraj.back();
        Action action;
        Dtype value;
        policy_->forward(state, action);
        qfunction_target_->forward(state, action, value);
        parTrajSet_[i].updateValueTrajWithNewTermValue(value, task_[0]->discountFtr());
      }
    }

    /// update values of the best trajectories
    for (int i = 0; i < bestTrajSet_.size(); i++) {
      if (bestTrajSet_[i].termType == TerminationType::terminalState)
        continue;

      State state = bestTrajSet_[i].stateTraj.back();
      Action action;
      Dtype value;
      policy_->forward(state, action);
      qfunction_target_->forward(state, action, value);
      bestTrajSet_[i].updateValueTrajWithNewTermValue(value, task_[0]->discountFtr());
    }

    /// appending new trajectories
    if (parTrajSet_.size() + bestTrajSet_.size() <= trajectoryMemorySize_) {
      bestTrajSet_.insert(bestTrajSet_.end(), parTrajSet_.begin(), parTrajSet_.end());
    } else {
      unsigned totalTrajcN = parTrajSet_.size() + bestTrajSet_.size();
      std::vector<Dtype> score(totalTrajcN);
      std::vector<unsigned> indx(totalTrajcN);
      for (unsigned i = 0; i < totalTrajcN; i++) indx[i] = i;
      for (unsigned i = 0; i < bestTrajSet_.size(); i++)
        score[i] = bestTrajSet_[i].getAverageValue();
      for (unsigned i = bestTrajSet_.size(); i < totalTrajcN; i++)
        score[i] = parTrajSet_[i - bestTrajSet_.size()].getAverageValue();
      Math::MathFunc::sort(score, indx);
      std::vector<Trajectory> bestTrajCpy(bestTrajSet_);
      for (int i = 0; i < bestTrajSet_.size(); i++)
        bestTrajSet_[i] = (indx[i] < trajectoryMemorySize_) ? bestTrajCpy[i] : parTrajSet_[indx[i] - trajectoryMemorySize_];
    }
    Utils::timer->stopTimer("Computing values and sorting");

    Utils::timer->startTimer("Qfunction and Policy update");
    if ( trajectoryN > trajectoryMemorySize_ )
      for(int i=0; i< nnUpdatePerIter_; i++)
        updateQfunctionAndPolicy();
    Utils::timer->stopTimer("Qfunction and Policy update");
    trajectoryN += task_.size();
  }

  void updateQfunctionAndPolicy() {
    StateBatch state_t(StateDim, batSize_), state_tp1(StateDim, batSize_);
    ActionBatch action_t(ActionDim, batSize_), action_tp1(ActionDim, batSize_);
    ValueBatch cost_t(batSize_), value_tp1(batSize_), value_t(batSize_);
    ScalarBatch termType(batSize_);
    Dtype termValue = task_[0]->termValue();
    Dtype disFtr = task_[0]->discountFtr();

    memorySARS_->sampleRandomBatch(state_t, action_t, cost_t, state_tp1, termType);

    ///// DDPG
    Utils::timer->startTimer("Qfunction update");
    policy_target_->forward(state_tp1, action_tp1);
    qfunction_target_->forward(state_tp1, action_tp1, value_tp1);
    for (unsigned tupleID = 0; tupleID < batSize_; tupleID++)
      if (TerminationType(termType(tupleID)) == TerminationType::terminalState)
        value_tp1(tupleID) = termValue;

    for (unsigned tupleID = 0; tupleID < batSize_; tupleID++)
      value_t(tupleID) = cost_t(tupleID) + disFtr * value_tp1(tupleID);

    qfunction_->performOneSolverIter(state_t, action_t, value_t);
    Utils::timer->stopTimer("Qfunction update");

    Utils::timer->startTimer("Policy update");
    policy_->backwardUsingCritic(qfunction_, state_t);
    Utils::timer->stopTimer("Policy update");

    Utils::timer->startTimer("Target update");
    qfunction_target_->interpolateAPWith(qfunction_, tau_);
    policy_target_->interpolateAPWith(policy_, tau_);
    Utils::timer->stopTimer("Target update");
  }

  /////////////////////////// Core ///////////////////////////////
  std::vector<Task_ *> task_;
  std::vector<Memory::Trajectory<Dtype, StateDim, ActionDim> > testTraj_;
  Qfunction_ *qfunction_, *qfunction_target_;
  Policy_ *policy_, *policy_target_;
  Noise::NoNoise<Dtype, ActionDim> noNoise_;
  Acquisitor_ *acquisitor_;
  ReplayMemory_ *memorySARS_;
  std::vector<Noise_ *> noise_;
  std::vector<Trajectory> parTrajSet_;
  std::vector<Trajectory> bestTrajSet_;
  unsigned trajectoryMemorySize_ = 0;
  unsigned trajectoryN = 0;
  std::vector<std::pair<unsigned, unsigned> > preTrajId_;
  RandomNumberGenerator<Dtype> rn_;
  unsigned batSize_;
  unsigned nnUpdatePerIter_;
  Dtype tau_;
  VectorXD parameter_;
  Dtype trajTimeLimit_;

  /////////////////////////// visualization
  int vis_lv_ = 0;

  /////////////////////////// logging
  unsigned long int numOfStepsTaken_ = 0;

  /////////////////////////// testing
  unsigned testingTrajN_;
  TestAcquisitor_ testAcquisitor_;

};

}
}

#endif //RAI_DDTS_HPP
