//
// Created by joonho on 11/21/17.
//

#ifndef RAI_RDPG_HPP
#define RAI_RDPG_HPP

#include <iostream>
#include "glog/logging.h"

#include "rai/tasks/common/Task.hpp"
#include <Eigen/Core>
#include <rai/noiseModel/NoNoise.hpp>
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
#include "rai/experienceAcquisitor/TrajectoryAcquisitor_Parallel.hpp"
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
  typedef rai::Algorithm::LearningData<Dtype, StateDim, ActionDim> Dataset;

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
       unsigned n_epoch,
       unsigned n_newEpisodesPerEpoch,
       unsigned batchSize,
       unsigned testingTrajN,
       Dtype maxGradNorm = 0.1,
       Dtype tau = 1e-3):
      qfunction_(qfunction),
      qfunction_target_(qfunction_target),
      policy_(policy),
      policy_target_(policy_target),
      noise_(noise),
      acquisitor_(acquisitor),
      memory(memory),
      n_epoch_(n_epoch),
      n_newEpisodesPerEpoch_(n_newEpisodesPerEpoch),
      batSize_(batchSize),
      tau_(tau),
      testingTrajN_(testingTrajN),
      task_(task),
      maxGradNorm_(maxGradNorm),
      Dataset_(false,true){

    ///Construct Dataset
    timeLimit = task_[0]->timeLimit();

    for (int i = 0; i < task_.size(); i++)
      noiseBasePtr_.push_back(noise_[i]);

    policy_->copyAPFrom(policy_target_);
    qfunction_->copyAPFrom(qfunction_target_);
  };
  ~RDPG(){};

  void initiallyFillTheMemory() {
    LOG(INFO) << "FillingMemory" ;
    if (vis_lv_ > 1) task_[0]->turnOnVisualization("");
    acquisitor_->acquireNEpisodes(task_, noiseBasePtr_, policy_, memory->getCapacity());
    if (vis_lv_ > 1) task_[0]->turnOffVisualization();

    timer->startTimer("SavingHistory");
    memory->SaveHistory(acquisitor_->traj);
    timer->stopTimer("SavingHistory");
    LOG(INFO) << "Done" ;
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
    if(numOfEpisodes > n_newEpisodesPerEpoch_) numOfEpisodes = n_newEpisodesPerEpoch_;

    for (unsigned i = 0; i < numOfEpisodes/n_newEpisodesPerEpoch_; i++)
      learnForOneCycle();
  }

  void setVisualizationLevel(int vis_lv) { vis_lv_ = vis_lv; }

 private:

  void learnForOneCycle() {
    if (vis_lv_ > 1) task_[0]->turnOnVisualization("");
    acquisitor_->acquireNEpisodes(task_, noiseBasePtr_, policy_, 1);
    if (vis_lv_ > 1) task_[0]->turnOffVisualization();
    timer->startTimer("SavingHistory");
    memory->SaveHistory(acquisitor_->traj);
    timer->stopTimer("SavingHistory");

    Utils::timer->startTimer("Qfunction and Policy update");
    for (unsigned i = 0; i <  n_epoch_; i++)
      updateQfunctionAndPolicy();
    Utils::timer->stopTimer("Qfunction and Policy update");
  }

  void updateQfunctionAndPolicy() {
    Dtype termValue = task_[0]->termValue();
    Dtype disFtr = task_[0]->discountFtr();

    timer->startTimer("SamplingHistory");
    memory->sampleRandomHistory(Dataset_, batSize_);
    timer->stopTimer("SamplingHistory");

    Utils::timer->startTimer("Qfunction update");
    ///Target
    Tensor<Dtype, 3> value_;
    Tensor<Dtype, 3> value_target("targetQValue");
    Tensor<Dtype, 3> action_target({ActionDim, Dataset_.maxLen, batSize_},"sampledAction");

    value_.resize(1, Dataset_.maxLen, batSize_);
    value_target.resize(1, Dataset_.maxLen, batSize_);
    value_target.setZero();

    policy_target_->forward(Dataset_.states, action_target);
    qfunction_target_->forward(Dataset_.states, action_target, value_);

    for (unsigned batchID = 0; batchID < batSize_; batchID++) {
      if (TerminationType(Dataset_.termtypes[batchID]) == TerminationType::terminalState)
        value_target.eTensor()(0, Dataset_.lengths[batchID]-1 , batchID) = termValue; ///last elem for each target batch
    }
    for (unsigned batchID = 0; batchID < batSize_; batchID++){
      for(unsigned timeID = 0; timeID< Dataset_.lengths[batchID] - 2 ; timeID++)
        value_target.eTensor()(0,timeID,batchID) = Dataset_.costs.eTensor()(timeID,batchID)  + disFtr * value_.eTensor()(0,timeID+1,batchID) ;
    }

    qfunction_->performOneSolverIter(&Dataset_, value_target);
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
  Dataset Dataset_;

  /////////////////////////// Algorithmic parameter ///////////////////
  double timeLimit;
  unsigned batSize_;
  Dtype tau_;
  Dtype maxGradNorm_;
  unsigned n_epoch_;
  unsigned n_newEpisodesPerEpoch_;

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
