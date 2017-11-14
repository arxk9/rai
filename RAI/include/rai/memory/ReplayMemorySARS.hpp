/*
 * ReplayMemory.hpp
 *
 *  Created on: Mar 28, 2016
 *      Author: jemin
 */

#ifndef ReplayMemorySARS_HPP_
#define ReplayMemorySARS_HPP_

// Eigen
#include <Eigen/Dense>
#include <Eigen/Core>
#include <mutex>
#include <algorithm>
#include "raiCommon/utils/RandomNumberGenerator.hpp"
#include "glog/logging.h"
#include "raiCommon/enumeration.hpp"

namespace rai {
namespace Memory {

template<typename Dtype, int stateDimension, int actionDimension>
class ReplayMemorySARS {

  typedef Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic> MatrixXD;
  typedef Eigen::Matrix<Dtype, stateDimension, Eigen::Dynamic> StateBatch;
  typedef Eigen::Matrix<Dtype, actionDimension, Eigen::Dynamic> ActionBatch;
  typedef Eigen::Matrix<Dtype, 1, Eigen::Dynamic> ScalarBatch;
  typedef Eigen::Matrix<Dtype, stateDimension, 1> State;
  typedef Eigen::Matrix<Dtype, actionDimension, 1> Action;

 public:

  ReplayMemorySARS(unsigned capacity) :
      size_(0), memoryIdx_(0) {
    state_t_ = new StateBatch(stateDimension, capacity);
    state_tp1_ = new StateBatch(stateDimension, capacity);
    action_t_ = new ActionBatch(actionDimension, capacity);
    cost_ = new ScalarBatch(1, capacity);
    terminationFlag_ = new ScalarBatch(1, capacity);
    capacity_ = capacity;
  }

  ~ReplayMemorySARS() {
    delete state_t_;
    delete state_tp1_;
    delete action_t_;
    delete cost_;
    delete terminationFlag_;
  }

  inline void saveAnExperienceTuple(State &state_t,
                                    Action &action_t,
                                    Dtype cost,
                                    State &state_tp1,
                                    TerminationType termType) {
    std::lock_guard<std::mutex> lockModel(memoryMutex_);
    state_t_->col(memoryIdx_) = state_t;
    state_tp1_->col(memoryIdx_) = state_tp1;
    action_t_->col(memoryIdx_) = action_t;
    (*cost_)(memoryIdx_) = cost;
    (*terminationFlag_)(memoryIdx_) = Dtype(termType);
    memoryIdx_ = (memoryIdx_ + 1) % capacity_;
    size_++;
    size_ = std::min(size_, capacity_);
  }

  inline void sampleRandomBatch(StateBatch &state_t_batch,
                                ActionBatch &action_t_batch,
                                ScalarBatch &cost_batch,
                                StateBatch &state_tp1_batch,
                                ScalarBatch &terminationFlag_tp1_batch) {
    int batchSize = state_t_batch.cols();
    LOG_IF(FATAL, size_ < batchSize * 1.2) <<
                                           "You don't have enough memories in the storage! accumulate more memories before you update your Q-function";
    unsigned memoryIdx[batchSize];
    ///// randomly sampling memory indeces
    for (unsigned i = 0; i < batchSize; i++) {
      memoryIdx[i] = rn_.intRand(0, size_ - 1);
      for (unsigned j = 0; j < i; j++) {
        if (memoryIdx[i] == memoryIdx[j]) {
          i--;
          break;
        }
      }
    }
    ///// saving memory to the batch
    for (unsigned i = 0; i < batchSize; i++) {
      state_t_batch.col(i) = state_t_->col(memoryIdx[i]);
      state_tp1_batch.col(i) = state_tp1_->col(memoryIdx[i]);
      action_t_batch.col(i) = action_t_->col(memoryIdx[i]);
      cost_batch.col(i) = cost_->col(memoryIdx[i]);
      terminationFlag_tp1_batch.col(i) = terminationFlag_->col(memoryIdx[i]);
    }
  }

  inline void sampleRandomBatch(ReplayMemorySARS &batchMemory) {
    unsigned batchSize = batchMemory.getSize();
    LOG_IF(FATAL, size_ < batchSize) <<
                                     "You don't have enough memories in the storage! accumulate more memories before you update your Q-function";
    unsigned memoryIdx[batchSize];
    ///// randomly sampling memory indeces
    for (unsigned i = 0; i < batchSize; i++) {
      memoryIdx[i] = rn_.intRand(0, size_ - 1);
      for (unsigned j = 0; j < i; j++) {
        if (memoryIdx[i] == memoryIdx[j]) {
          i--;
          break;
        }
      }
    }

    ///// saving memory to the batch
    for (unsigned i = 0; i < batchSize; i++) {
      State state = state_t_->col(memoryIdx[i]);
      State state_tp1 = state_tp1_->col(memoryIdx[i]);
      Action action = action_t_->col(memoryIdx[i]);
      Dtype cost = cost_->data()[memoryIdx[i]];
      TerminationType termination = TerminationType(terminationFlag_->data()[memoryIdx[i]]);
      batchMemory.saveAnExperienceTuple(state, action, cost, state_tp1, termination);
    }
  }

  inline void sampleRandomStateBatch(StateBatch &state_t_batch) {
    unsigned batchSize = state_t_batch.cols();
    LOG_IF(FATAL, size_ < batchSize) <<
                                     "You don't have enough memories in the storage! accumulate more memories before you update your Q-function";
    unsigned memoryIdx[batchSize];
    ///// randomly sampling memory indeces
    for (unsigned i = 0; i < batchSize; i++) {
      memoryIdx[i] = rn_.intRand(0, size_ - 1);
      for (unsigned j = 0; j < i; j++) {
        if (memoryIdx[i] == memoryIdx[j]) {
          i--;
          break;
        }
      }
    }

    ///// saving memory to the batch
    for (unsigned i = 0; i < batchSize; i++) {
      State state = state_t_->col(memoryIdx[i]);
      state_t_batch.col(i) = state;
    }
  }

  inline void clearTheMemoryAndSetNewBatchSize(int newMemorySize) {
    std::lock_guard<std::mutex> lockModel(memoryMutex_);
    delete state_t_;
    delete state_tp1_;
    delete action_t_;
    delete cost_;
    delete terminationFlag_;
    state_t_ = new StateBatch(stateDimension, newMemorySize);
    state_tp1_ = new StateBatch(stateDimension, newMemorySize);
    action_t_ = new ActionBatch(actionDimension, newMemorySize);
    cost_ = new ScalarBatch(1, newMemorySize);
    terminationFlag_ = new ScalarBatch(1, newMemorySize);

    size_ = 0;
    memoryIdx_ = 0;
    capacity_ = newMemorySize;
  }

  inline void saveAnExperienceTupleWithSparcification_DiagonalMetric(State &state_t,
                                                                     Action &action_t,
                                                                     Dtype cost,
                                                                     State &state_tp1,
                                                                     TerminationType termType,
                                                                     State &stateMetricInverse,
                                                                     Action &actionMetricInverse,
                                                                     Dtype threshold) {
    bool saved = true;
    for (unsigned memoryID = 0; memoryID < size_; memoryID++) {
      auto diff_state = state_t_->col(memoryID) - state_t;
      auto diff_action = action_t_->col(memoryID) - action_t;
      auto dist = sqrt(diff_state.cwiseProduct(diff_state).dot(stateMetricInverse) +
          diff_action.cwiseProduct(diff_action).dot(actionMetricInverse));

      if (dist < threshold) {
        saved = false;
        break;
      }
    }

    if (saved)
      saveAnExperienceTuple(state_t, action_t, cost, state_tp1, termType);
  }

  Dtype getDist2ClosestSample(State &state_t,
                              Action &action_t,
                              State &stateMetricInverse,
                              Action &actionMetricInverse) {
    Dtype dist, closest_dist = 1e99;
    for (unsigned memoryID = 0; memoryID < size_; memoryID++) {
      dist = sqrt((state_t_->col(memoryID) - state_t).squaredNorm()
                      + (action_t_->col(memoryID) - action_t).squaredNorm());
      if (dist < closest_dist)
        closest_dist = dist;
    }
    return closest_dist;
  }

  inline typename StateBatch::ColsBlockXpr getState_t() {
    return state_t_->leftCols(size_);
  }

  inline typename StateBatch::ColsBlockXpr getState_tp1() {
    return state_tp1_->leftCols(size_);
  }

  inline typename ActionBatch::ColsBlockXpr getAction_t() {
    return action_t_->leftCols(size_);
  }

  inline typename ScalarBatch::ColsBlockXpr getCost_() {
    return cost_->leftCols(size_);
  }

  inline typename ScalarBatch::ColsBlockXpr getTeminationFlag() {
    return terminationFlag_->leftCols(size_);
  }

  unsigned getCapacity() {
    return capacity_;
  }

  unsigned getSize() {
    return size_;
  }

  void printOutMemory() {
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "--------------Replay memory printout" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "state_t_" << std::endl << state_t_->leftCols(size_) << std::endl;
    std::cout << "action_t_" << std::endl << action_t_->leftCols(size_) << std::endl;
    std::cout << "cost_" << std::endl << cost_->leftCols(size_) << std::endl;
    std::cout << "terminationFlag_" << std::endl << terminationFlag_->leftCols(size_) << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
  }

 private:
  StateBatch *state_t_;
  StateBatch *state_tp1_;
  ActionBatch *action_t_;
  ScalarBatch *cost_;
  ScalarBatch *terminationFlag_;
  unsigned size_;
  unsigned memoryIdx_;
  unsigned capacity_;

 private:
  static std::mutex memoryMutex_;
  RandomNumberGenerator <Dtype> rn_;
};
}
}

template<typename Dtype, int stateDimension, int actionDimension>
std::mutex rai::Memory::ReplayMemorySARS<Dtype, stateDimension, actionDimension>::memoryMutex_;

#endif /* ReplayMemorySARS_HPP_ */
