//
// Created by joonho on 11/17/17.
//

#ifndef RAI_REPLAYMEMORYTRAJECTORY_HPP
#define RAI_REPLAYMEMORYTRAJECTORY_HPP

// Eigen
#include <Eigen/Dense>
#include <Eigen/Core>
#include <mutex>
#include <algorithm>
#include "raiCommon/utils/RandomNumberGenerator.hpp"
#include "glog/logging.h"
#include "raiCommon/enumeration.hpp"
#include <rai/RAI_Tensor.hpp>
#include "rai/memory/Trajectory.hpp"

namespace rai {
namespace Memory {

template<typename Dtype, int stateDimension, int actionDimension>
class ReplayMemoryHistory{

  typedef rai::Tensor<Dtype, 3> Tensor3D;
  typedef rai::Tensor<Dtype, 2> Tensor2D;
  typedef rai::Tensor<Dtype, 1> Tensor1D;
  typedef Eigen::Matrix<Dtype, 1, Eigen::Dynamic> ScalarBatch;
  typedef Eigen::Matrix<Dtype, stateDimension, 1> State;
  typedef Eigen::Matrix<Dtype, actionDimension, 1> Action;


 public:

  ReplayMemoryHistory(unsigned capacity, bool distInfo = true) : size_(0), batchIdx_(0),distInfo_(distInfo) {
    stateTensor_ = new Tensor3D;
    actionTensor_  = new Tensor3D;
    costTensor_ = new Tensor2D;
    len_ = new Tensor1D;
    if(distInfo_){
      actionNoiseTensor_ = new Tensor3D;
      stdevs_ = new ScalarBatch(1,capacity);
    }

    termtypes_ = new Tensor1D;
    capacity_ = capacity;
  }

  ~ReplayMemoryHistory() {
    delete stateTensor_;
    delete actionTensor_;
    delete costTensor_;
    delete len_;
    delete termtypes_;
    if(distInfo_){
      delete actionNoiseTensor_;
      delete stdevs_;
    }

  }


  inline void SaveHistory(Tensor3D &states,
                          Tensor3D &actions,
                          Tensor2D &costs,
                          Tensor1D &lengths,
                          Tensor1D &termTypes)
  {
    int margin = size_ - batchIdx_;
    maxlen_ = std::max(states.cols(),maxlen_);

    if(stateTensor_->cols() < maxlen_)
    {

    }
    std::lock_guard<std::mutex> lockModel(memoryMutex_);
    if(margin < states.batches())
    {
      stateTensor_->batchBlock(batchIdx_,margin) = states.batchBlock(0,margin);
      actionTensor_->batchBlock(batchIdx_,margin) = actions.batchBlock(0,margin);
      costTensor_->block(0,batchIdx_,maxlen_,margin) = costs.block(0,0,maxlen_,margin);
      len_->block(batchIdx_,margin) = lengths.block(0,margin);
      termtypes_->block(batchIdx_,margin) = termTypes.block(0,margin);

      stateTensor_->batchBlock(0,states.batches()-margin) = states.batchBlock(margin,states.batches()-margin);
      actionTensor_->batchBlock(0,states.batches()-margin) = actions.batchBlock(margin,states.batches()-margin);
      costTensor_->block(0,0,maxlen_,states.batches()-margin) = costs.block(0,margin,maxlen_,states.batches()-margin);
      len_->block(0,states.batches()-margin) = lengths.block(margin,states.batches()-margin);
      termtypes_->block(0,states.batches()-margin) = termTypes.block(margin,states.batches()-margin);
    }
//    state_t_->col(memoryIdx_) = state_t;
//    state_tp1_->col(memoryIdx_) = state_tp1;
//    action_t_->col(memoryIdx_) = action_t;
//    (*cost_)(memoryIdx_) = cost;
//    (*terminationFlag_)(memoryIdx_) = Dtype(termType);
//    memoryIdx_ = (memoryIdx_ + 1) % capacity_;
//    size_++;
//    size_ = std::min(size_, capacity_);
  }

//
//  inline void clearTheMemoryAndSetNewBatchSize(int newMemorySize) {
//    std::lock_guard<std::mutex> lockModel(memoryMutex_);
//    delete state_t_;
//    delete state_tp1_;
//    delete action_t_;
//    delete cost_;
//    delete terminationFlag_;
//    state_t_ = new StateBatch(stateDimension, newMemorySize);
//    state_tp1_ = new StateBatch(stateDimension, newMemorySize);
//    action_t_ = new ActionBatch(actionDimension, newMemorySize);
//    cost_ = new ScalarBatch(1, newMemorySize);
//    terminationFlag_ = new ScalarBatch(1, newMemorySize);
//
//    size_ = 0;
//    memoryIdx_ = 0;
//    capacity_ = newMemorySize;
//  }
//
//
//
//  inline void sampleRandomStateBatch(StateBatch &state_t_batch) {
//    unsigned batchSize = state_t_batch.cols();
//    LOG_IF(FATAL, size_ < batchSize) <<
//                                     "You don't have enough memories in the storage! accumulate more memories before you update your Q-function";
//    unsigned memoryIdx[batchSize];
//    ///// randomly sampling memory indeces
//    for (unsigned i = 0; i < batchSize; i++) {
//      memoryIdx[i] = rn_.intRand(0, size_ - 1);
//      for (unsigned j = 0; j < i; j++) {
//        if (memoryIdx[i] == memoryIdx[j]) {
//          i--;
//          break;
//        }
//      }
//    }
//
//    ///// saving memory to the batch
//    for (unsigned i = 0; i < batchSize; i++) {
//      State state = state_t_->col(memoryIdx[i]);
//      state_t_batch.col(i) = state;
//    }
//  }
 private:
  Tensor3D* stateTensor_;
  Tensor3D* actionTensor_;
  Tensor3D* actionNoiseTensor_;
  Tensor2D* costTensor_;
  Tensor1D* len_;
  Tensor1D* termtypes_;

  ScalarBatch* stdevs_;

  bool distInfo_;
  unsigned size_;
  unsigned maxlen_;
  unsigned batchIdx_;
  unsigned capacity_;
  static std::mutex memoryMutex_;
  RandomNumberGenerator <Dtype> rn_;
};
}
}

template<typename Dtype, int stateDimension, int actionDimension>
std::mutex rai::Memory::ReplayMemoryHistory<Dtype, stateDimension, actionDimension>::memoryMutex_;

#endif //RAI_REPLAYMEMORYTRAJECTORY_HPP
