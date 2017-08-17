//
// Created by jhwangbo on 12.09.16.
//

#ifndef RAI_TRUSTREGIONPOLICYOPTIMIZATION_HPP
#define RAI_TRUSTREGIONPOLICYOPTIMIZATION_HPP

#include <iostream>

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
#include <rai/RAI_core>
#include <rai/experienceAcquisitor/TrajectoryAcquisitor_SingleThreadBatch.hpp>
#include <rai/experienceAcquisitor/TrajectoryAcquisitor_Sequential.hpp>

// Neural network
//function approximations
#include "rai/function/common/DeterministicPolicy.hpp"
#include "rai/function/common/Qfunction.hpp"

// common
#include "enumeration.hpp"
#include "math/inverseUsingCholesky.hpp"
#include "math/ConjugateGradient.hpp"
#include "math.h"
#include "rai/RAI_core"

namespace RAI {
namespace Algorithm {

template<typename Dtype, int StateDim, int ActionDim>
class DDPG {

 public:

  typedef Eigen::Matrix<Dtype, StateDim, 1> State;
  typedef Eigen::Matrix<Dtype, StateDim, Eigen::Dynamic> StateBatch;
  typedef Eigen::Matrix<Dtype, ActionDim, 1> Action;
  typedef Eigen::Matrix<Dtype, ActionDim, Eigen::Dynamic> ActionBatch;
  typedef Eigen::Matrix<Dtype, 1, Eigen::Dynamic> ValueBatch;
  typedef Eigen::Matrix<Dtype, 1, Eigen::Dynamic> ScalarBatch;
  typedef Eigen::Matrix<Dtype, StateDim, ActionDim> JacobianStateResAct;
  typedef Eigen::Matrix<Dtype, 1, ActionDim> JacobianCostResAct;
  typedef Eigen::Matrix<Dtype, -1, -1> MatrixXD;
  typedef Eigen::Matrix<Dtype, -1, 1> VectorXD;
  typedef Eigen::Matrix<Dtype, 1, -1> RowVectorXD;

  using Task_ = Task::Task<Dtype, StateDim, ActionDim, 0>;
  using ReplayMemory_ = RAI::Memory::ReplayMemorySARS<Dtype, StateDim, ActionDim>;
  using Qfunction_ = FuncApprox::Qfunction<Dtype, StateDim, ActionDim>;
  using Policy_ = FuncApprox::DeterministicPolicy<Dtype, StateDim, ActionDim>;
  using Noise_ = Noise::Noise<Dtype, ActionDim>;
  using Trajectory = Memory::Trajectory<Dtype, StateDim, ActionDim>;
  using Acquisitor_ = ExpAcq::ExperienceTupleAcquisitor<Dtype, StateDim, ActionDim>;
  //using TestAcquisitor_ = ExpAcq::TrajectoryAcquisitor_MultiThreadBatch<Dtype, StateDim, ActionDim>;
  using TestAcquisitor_ = ExpAcq::TrajectoryAcquisitor_Sequential<Dtype, StateDim, ActionDim>;
 // using TestAcquisitor_ = ExpAcq::TrajectoryAcquisitor_SingleThreadBatch<Dtype, StateDim, ActionDim>;

  DDPG(std::vector<Task_ *> &task,
       Qfunction_ *qfunction,
       Qfunction_ *qfunction_target,
       Policy_ *policy,
       Policy_ *policy_target,
       std::vector<Noise_ *> &noise,
       Acquisitor_ *acquisitor,
       ReplayMemory_ *memory,
       unsigned batchSize,
       unsigned testingTrajN,  // how many trajectories to use to test the policy
       Dtype tau = 1e-3) :
      qfunction_(qfunction),
      qfunction_target_(qfunction_target),
      policy_(policy),
      policy_target_(policy_target),
      noise_(noise),
      acquisitor_(acquisitor),
      memorySARS_(memory),
      batSize_(batchSize),
      tau_(tau),
      testingTrajN_(testingTrajN),
      testTraj_(testingTrajN, Trajectory()),
      task_(task) {

    for (auto &task : task_)
      task->setToInitialState();
    Utils::logger->addVariableToLog(2, "Nominal performance", "");

    policy_->copyAPFrom(policy_target_);
    qfunction_->copyAPFrom(qfunction_target_);
  };

  ~DDPG() {};

  void initiallyFillTheMemory() {
    if (vis_lv_ > 1) task_[0]->turnOnVisualization("");
    Utils::timer->startTimer("Simulation");
    acquisitor_->acquire(task_, policy_, noise_, memorySARS_, batSize_ * 40);
    Utils::timer->stopTimer("Simulation");
    if (vis_lv_ > 1) task_[0]->turnOffVisualization();
  }

  void learnForNSteps(int numOfSteps) {
    iterNumber_++;

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
    std::vector<Noise_*> noiseVec;
    for (int i = 0; i < task_.size(); i++)
      noiseVec.push_back( &noNoises[i] );

    if (vis_lv_ > 0) {
      task_[0]->turnOnVisualization("");
      if (task_[0]->shouldRecordVideo())
        task_[0]->startRecordingVideo(RAI_LOG_PATH + "/" + std::to_string(iterNumber_), "nominalPolicy");
    }
    Dtype averageCost = testAcquisitor_.acquire( task_,
                                                 policy_,
                                                 noiseVec,
                                                 testTraj_,
                                                 startState,
                                                 task_[0]->timeLimit(), false );
    if (vis_lv_ > 0) task_[0]->turnOffVisualization();
    if (task_[0]->shouldRecordVideo()) { task_[0]->endRecordingVideo(); }
    Utils::logger->appendData("Nominal performance",
                              acquisitor_->stepsTaken(),
                              averageCost);

    LOG(INFO) << "steps taken " << logger->getData("Nominal performance")->at(0).back()
              << ", average cost " << logger->getData("Nominal performance")->at(1).back();
    /// reset all for learning
    for (auto &task : task_)
      task->setToInitialState();
    for (auto &noise : noise_)
      noise->initializeNoise();

    Utils::timer->enable();
    /////////////////////////////////////////////////////////////////////////
    for (unsigned i = 0; i < numOfSteps / task_.size(); i++)
      learnForOneStep();
  }

  void setVisualizationLevel(int vis_lv) { vis_lv_ = vis_lv; }

 private:

  void sampleBatchOfInitial(StateBatch& initial){
    for (int trajID = 0; trajID < initial.cols(); trajID++) {
      State state;
      task_[0]->setToInitialState();
      task_[0]->getState(state);
      initial.col(trajID) = state;
    }
  }

  void learnForOneStep() {
    Utils::timer->startTimer("Simulation");
    if (vis_lv_ > 1) task_[0]->turnOnVisualization("");
    acquisitor_->acquire(task_, policy_, noise_, memorySARS_, task_.size());
    if (vis_lv_ > 1) task_[0]->turnOffVisualization();
    Utils::timer->stopTimer("Simulation");
    numOfStepsTaken_ += task_.size();
    Utils::timer->startTimer("Qfunction and Policy update");
    updateQfunctionAndPolicy();
    Utils::timer->stopTimer("Qfunction and Policy update");
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

  unsigned batSize_;
  Dtype tau_;

  /////////////////////////// visualization
  int vis_lv_ = 0;

  /////////////////////////// logging
  unsigned long int numOfStepsTaken_ = 0;

  /////////////////////////// testing
  unsigned testingTrajN_;
  TestAcquisitor_ testAcquisitor_;
  int iterNumber_ = 0;
};

}
}

#endif //RAI_TRUSTREGIONPOLICYOPTIMIZATION_HPP
