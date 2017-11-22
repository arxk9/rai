//
// Created by joonho on 11/21/17.
//

#ifndef RAI_RDPG_HPP
#define RAI_RDPG_HPP

#include <iostream>
#include "glog/logging.h"

#include "rai/tasks/common/Task.hpp"
#include <Eigen/Core>
#include <rai/noiseModel/OrnsteinUhlenbeckNoise.hpp>
#include <rai/noiseModel/NoNoise.hpp>
#include <Eigen/Cholesky>
#include <Eigen/Jacobi>
#include <Eigen/Cholesky>
#include <boost/bind.hpp>
#include <math.h>
#include "rai/RAI_core"
#include <vector>

#include "rai/function/common/Policy.hpp"
#include "rai/function/common/ValueFunction.hpp"
#include "rai/function/common/StochasticPolicy.hpp"
#include "rai/common/VectorHelper.hpp"

// memory
#include "rai/memory/Trajectory.hpp"
#include "rai/memory/ReplayMemoryHistory.hpp"

// acquisitor
#include "rai/experienceAcquisitor/TrajectoryAcquisitor_MultiThreadBatch.hpp"
#include <rai/experienceAcquisitor/TrajectoryAcquisitor_SingleThreadBatch.hpp>
#include <rai/experienceAcquisitor/TrajectoryAcquisitor_Sequential.hpp>
#include <rai/algorithm/common/LearningData.hpp>
#include <rai/function/tensorflow/RecurrentQfunction_TensorFlow.hpp>
#include <rai/function/tensorflow/RecurrentDeterministicPolicy_Tensorflow.hpp>

// common
#include "raiCommon/enumeration.hpp"
#include "common/PerformanceTester.hpp"

namespace rai {
namespace Algorithm {

template<typename Dtype, int StateDim, int ActionDim>
class RDPG {

 public:

  typedef Eigen::Matrix<Dtype, StateDim, 1> State;
  typedef Eigen::Matrix<Dtype, StateDim, Eigen::Dynamic> StateBatch;
  typedef Eigen::Matrix<Dtype, ActionDim, 1> Action;
  typedef Eigen::Matrix<Dtype, 1, 1> Value;
  typedef Eigen::Matrix<Dtype, 1, Eigen::Dynamic> ValueBatch;
  typedef Eigen::Matrix<Dtype, ActionDim, ActionDim> Covariance;
  typedef Eigen::Matrix<Dtype, -1, -1> MatrixXD;
  typedef Eigen::Matrix<Dtype, -1, 1> VectorXD;
  typedef Eigen::Matrix<Dtype, -1, 1> Parameter;
  typedef rai::Algorithm::history<Dtype, StateDim, ActionDim> TensorBatch_;

  using Task_ = Task::Task<Dtype, StateDim, ActionDim, 0>;
  using Noise_ = Noise::Noise<Dtype, ActionDim>;

  using Qfunction_ = FuncApprox::RecurrentQfunction_TensorFlow<Dtype, StateDim, ActionDim>;
  using Policy_ = FuncApprox::RecurrentDeterministicPolicy_TensorFlow<Dtype, StateDim, ActionDim>;
  using Acquisitor_ = ExpAcq::TrajectoryAcquisitor<Dtype, StateDim, ActionDim>;
  using TestAcquisitor_ = ExpAcq::TrajectoryAcquisitor_Sequential<Dtype, StateDim, ActionDim>;
  using ReplayMemory_ = rai::Memory::ReplayMemoryHistory<Dtype, StateDim, ActionDim>;

  RDPG(std::vector<Task_ *> &task,
       Qfunction_ *qfunction,
       Qfunction_ *qfunction_target,
       Policy_ *policy,
       Policy_ *policy_target,
       std::vector<Noise_ *> &noise,
       Acquisitor_ *acquisitor,
       ReplayMemory_ *memory,
       unsigned batchSize,
       unsigned testingTrajN,
       int n_epoch = 30,
       int minibatchSize = 0,
       Dtype tau = 1e-3):
      qfunction_(qfunction),
      qfunction_target_(qfunction_target),
      policy_(policy),
      policy_target_(policy_target),
      noise_(noise),
      acquisitor_(acquisitor),
      memory(memory),
      batSize_(batchSize),
      tau_(tau),
      testingTrajN_(testingTrajN),
      task_(task),
      ld_(acquisitor) {

    ///Construct minibatch
    Dataset_.minibatch = new TensorBatch_;

    parameter_.setZero(policy_->getLPSize());
    policy_->getLP(parameter_);

    timeLimit = task_[0]->timeLimit();

    for (int i = 0; i < task_.size(); i++)
      noiseBasePtr_.push_back(noise_[i]);

    policy_->copyAPFrom(policy_target_);
    qfunction_->copyAPFrom(qfunction_target_);
  };

  ~RDPG() {delete Dataset_.minibatch; };

  void initiallyFillTheMemory() {
    if (vis_lv_ > 1) task_[0]->turnOnVisualization("");
    Utils::timer->startTimer("Simulation");
    ld_.acquireNEpisodes(task_, noiseBasePtr_, policy_, memory->getCapacity());
    Utils::timer->stopTimer("Simulation");
    if (vis_lv_ > 1) task_[0]->turnOffVisualization();
    memory->SaveHistory(Dataset_.states, Dataset_.actions, Dataset_.costs, Dataset_.lengths, Dataset_.termtypes);
  }

  void learnForNepisodes(int numOfEpisodes) {
    iterNumber_++;

    //////////////// testing (not part of the algorithm) ////////////////////
    Utils::timer->disable();
    tester_.testPerformance(task_,
                            noise_,
                            policy_,
                            task_[0]->timeLimit(),
                            testingTrajN_,
                            acquisitor_->stepsTaken(),
                            vis_lv_,
                            std::to_string(iterNumber_));

    /// reset all for learning
    for (auto &task : task_)
      task->setToInitialState();
    for (auto &noise : noise_)
      noise->initializeNoise();

    Utils::timer->enable();
    /////////////////////////////////////////////////////////////////////////
    for (unsigned i = 0; i < numOfEpisodes; i++)
      learnForOneEpisode();
  }

  void setVisualizationLevel(int vis_lv) { vis_lv_ = vis_lv; }

 private:

  void sampleBatchOfInitial(StateBatch &initial) {
    for (int trajID = 0; trajID < initial.cols(); trajID++) {
      State state;
      task_[0]->setToInitialState();
      task_[0]->getState(state);
      initial.col(trajID) = state;
    }
  }

  void learnForOneEpisode() {
    Utils::timer->startTimer("Simulation");
    if (vis_lv_ > 1) task_[0]->turnOnVisualization("");
    ld_.acquireNEpisodes(task_, noiseBasePtr_, policy_, 1);
    if (vis_lv_ > 1) task_[0]->turnOffVisualization();
    Utils::timer->stopTimer("Simulation");
    memory->SaveHistory(Dataset_.states, Dataset_.actions, Dataset_.costs, Dataset_.lengths, Dataset_.termtypes);
    Utils::timer->startTimer("Qfunction and Policy update");
    updateQfunctionAndPolicy();
    Utils::timer->stopTimer("Qfunction and Policy update");
  }

  void updateQfunctionAndPolicy() {
    Dtype termValue = task_[0]->termValue();
    Dtype disFtr = task_[0]->discountFtr();

    Tensor<Dtype, 3> value_;
    Tensor<Dtype, 3> value_t("targetQValue");

    memory->sampleRandomHistory(Dataset_, batSize_);
    value_.resize(1, Dataset_.maxLen, Dataset_.batchNum);
    value_t.resize(1, Dataset_.maxLen, Dataset_.batchNum);

    Utils::timer->startTimer("Qfunction update");
    policy_target_->forward(Dataset_.states, Dataset_.actions);
    qfunction_target_->forward(Dataset_.states, Dataset_.actions, value_);

    for (unsigned batchID = 0; batchID < batSize_; batchID++) {
      if (TerminationType(Dataset_.termtypes[batchID]) == TerminationType::terminalState)
        value_.eTensor()(1, Dataset_.lengths[batchID], batchID) = termValue;
    }
    for (unsigned batchID = 0; batchID < batSize_; batchID++){
      for(unsigned timeID = 0; timeID<Dataset_.lengths[batchID]; timeID++)
      value_t.eTensor()(1,timeID,batchID) = Dataset_.costs.eTensor()(1,timeID,batchID)  + disFtr * value_t.eTensor()(1,timeID,batchID) ;
    }
//
//    for (unsigned tupleID = 0; tupleID < batSize_; tupleID++)
//      value_t(tupleID) = cost_t(tupleID) + disFtr * value_tp1(tupleID);

    qfunction_->performOneSolverIter(&Dataset_, value_t);
    Utils::timer->stopTimer("Qfunction update");

    Utils::timer->startTimer("Policy update");
    policy_->backwardUsingCritic(qfunction_, &Dataset_);
    Utils::timer->stopTimer("Policy update");

    Utils::timer->startTimer("Target update");
    qfunction_target_->interpolateAPWith(qfunction_, tau_);
    policy_target_->interpolateAPWith(policy_, tau_);
    Utils::timer->stopTimer("Target update");
  }

  /////////////////////////// Core //////////////////////////////////////////
  std::vector<Task_ *> task_;
  std::vector<Noise_ *> noise_;
  std::vector<Noise::Noise<Dtype, ActionDim> *> noiseBasePtr_;
  Qfunction_ *qfunction_, *qfunction_target_;
  Policy_ *policy_, *policy_target_;
  Acquisitor_ *acquisitor_;
  ReplayMemory_ *memory;
  PerformanceTester<Dtype, StateDim, ActionDim> tester_;
  LearningData<Dtype, StateDim, ActionDim> ld_;
  history<Dtype, StateDim, ActionDim> Dataset_;

  /////////////////////////// Algorithmic parameter ///////////////////
  double timeLimit;
  unsigned batSize_;
  Dtype tau_;

  /////////////////////////// batches
  ValueBatch advantage_, bellmanErr_;

  /////////////////////////// Policy parameter
  VectorXD parameter_;

  /////////////////////////// plotting
  int iterNumber_ = 0;

  ///////////////////////////testing
  unsigned testingTrajN_;

  /////////////////////////// visualization
  int vis_lv_ = 0;
};

}
}
#endif //RAI_RDPG_HPP
