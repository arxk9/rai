/*
 * ReplayMemoryS.hpp
 *
 *  Created on: Mar 28, 2016
 *      Author: jemin
 */

#ifndef ReplayMemoryS_HPP_
#define ReplayMemoryS_HPP_

// Eigen
#include <Eigen/Dense>
#include <Eigen/Core>
#include <mutex>
#include <algorithm>
#include "rai/common/math/RandomNumberGenerator.hpp"
#include "glog/logging.h"
#include "rai/common/enumeration.hpp"

namespace RAI {
namespace Memory {

template<typename Dtype, int stateDimension>
class ReplayMemoryS {

  typedef Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic> MatrixXD;
  typedef Eigen::Matrix<Dtype, stateDimension, Eigen::Dynamic> StateBatch;
  typedef Eigen::Matrix<Dtype, 1, Eigen::Dynamic> ScalarBatch;
  typedef Eigen::Matrix<Dtype, stateDimension, 1> State;

 public:

  ReplayMemoryS(int memorySize) :
      numberOfStoredTuples_(0), memoryIdx_(0) {
    state_t_ = new StateBatch(stateDimension, memorySize);
    memorySize_ = memorySize;
  }

  ~ReplayMemoryS() {
    delete state_t_;
  }

  void saveState(State &state_t) {
    state_t_->col(memoryIdx_) = state_t;
    memoryIdx_ = (memoryIdx_ + 1) % memorySize_;
    numberOfStoredTuples_++;
    numberOfStoredTuples_ = std::min(numberOfStoredTuples_, memorySize_);
  }

  void sampleRandomBatch(StateBatch &state_t_batch) {
    int batchSize = state_t_batch.cols();
    LOG_IF(FATAL, numberOfStoredTuples_ < batchSize * 1.2) <<
                                                           "You don't have enough memories in the storage! accumulate more memories before you update your Q-function";
    unsigned int memoryIdx[batchSize];
    ///// randomly sampling memory indeces
    for (int i = 0; i < batchSize; i++) {
      memoryIdx[i] = rn_.intRand(0, numberOfStoredTuples_ - 1);
      for (int j = 0; j < i; j++) {
        if (memoryIdx[i] == memoryIdx[j]) {
          i--;
          break;
        }
      }
    }

    ///// saving memory to the batch
    for (int i = 0; i < batchSize; i++) {
      state_t_batch.col(i) = state_t_->col(memoryIdx[i]);
    }
  }

  void sampleRandomBatch(ReplayMemoryS &batchMemory) {

    int batchSize = batchMemory.getMemorySize();
    LOG_IF(FATAL, numberOfStoredTuples_ < batchSize) <<
                                                     "You don't have enough memories in the storage! accumulate more memories before you update your Q-function";
    unsigned int memoryIdx[batchSize];
    ///// randomly sampling memory indeces
    for (int i = 0; i < batchSize; i++) {
      memoryIdx[i] = rn_.intRand(0, numberOfStoredTuples_ - 1);
      for (int j = 0; j < i; j++) {
        if (memoryIdx[i] == memoryIdx[j]) {
          i--;
          break;
        }
      }
    }

    ///// saving memory to the batch
    for (int i = 0; i < batchSize; i++) {
      State state = state_t_->col(memoryIdx[i]);
    }
  }

  void clearTheMemoryAndSetNewBatchSize(int newMemorySize) {
    delete state_t_;
    state_t_ = new StateBatch(newMemorySize);
    numberOfStoredTuples_ = 0;
    memoryIdx_ = 0;
    memorySize_ = newMemorySize;
  }

  void saveStateWithSparcification_DiagonalMetric(State &state_t,
                                                  State stateMetricInverse,
                                                  Dtype threshold) {
    bool saved = true;
    for (int memoryID = 0; memoryID < numberOfStoredTuples_; memoryID++) {
      auto diff_state = state_t_->col(memoryID) - state_t;
      auto dist = sqrt(diff_state.cwiseProduct(diff_state).dot(stateMetricInverse));

      if (dist < threshold) {
        saved = false;
        break;
      }
    }

    if (saved)
      saveState(state_t);
  }

  StateBatch *getState_t() {
    return state_t_;
  }

  int getMemorySize() {
    return memorySize_;
  }

  int getNumberOfStates() {
    return numberOfStoredTuples_;
  }

 private:
  StateBatch *state_t_;
  int numberOfStoredTuples_;
  int memoryIdx_;
  int memorySize_;

 private:
  static std::mutex memoryMutex_;
  RandomNumberGenerator<Dtype> rn_;
};
}
}

template<typename Dtype, int stateDimension>
std::mutex RAI::Memory::ReplayMemoryS<Dtype, stateDimension>::memoryMutex_;

#endif /* ReplayMemorySARS_HPP_ */
