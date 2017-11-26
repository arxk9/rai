//
// Created by jhwangbo on 08/08/17.
// This class's old data is deleted when you acquire new data
//

#ifndef RAI_LearningData_HPP
#define RAI_LearningData_HPP

#include <rai/memory/Trajectory.hpp>
#include <rai/RAI_core>
#include <rai/function/common/StochasticPolicy.hpp>
#include <rai/common/VectorHelper.hpp>
#include "rai/tasks/common/Task.hpp"

namespace rai {
namespace Algorithm {

template<typename Dtype, int StateDim, int ActionDim>
class LearningData {
  typedef Eigen::Matrix<Dtype, StateDim, 1> State;
  typedef Eigen::Matrix<Dtype, StateDim, Eigen::Dynamic> StateBatch;
  typedef Eigen::Matrix<Dtype, ActionDim, 1> Action;
  typedef Eigen::Matrix<Dtype, ActionDim, Eigen::Dynamic> ActionBatch;
  typedef Eigen::Matrix<Dtype, 1, 1> Value;
  typedef Eigen::Matrix<Dtype, 1, Eigen::Dynamic> ValueBatch;
  typedef Eigen::Matrix<Dtype, -1, -1> MatrixXD;
  typedef Eigen::Matrix<Dtype, -1, 1> VectorXD;
  typedef rai::Tensor<Dtype, 1> Tensor1D;
  typedef rai::Tensor<Dtype, 2> Tensor2D;
  typedef rai::Tensor<Dtype, 3> Tensor3D;

  using Trajectory_ = Memory::Trajectory<Dtype, StateDim, ActionDim>;
  using Task_ = Task::Task<Dtype, StateDim, ActionDim, 0>;

  LearningData() : maxLen(0), batchNum(0), batchID(0), miniBatch(nullptr) {
    states = "state";
    actions = "sampledAction";
    actionNoises = "actionNoise";

    costs = "costs";
    values = "value";
    advantages = "advantage";
//    stdevs = "stdevs";

    lengths = "length";
    termtypes = "termtypes";
  };
 private:
  virtual void fillminiBatch(int batchSize = 0) {

    if (miniBatch->batchNum != batchSize || miniBatch->maxLen != maxLen) {
      miniBatch->resize(maxLen, batchSize);
    }

    miniBatch->states = states.batchBlock(batchID, batchSize);
    miniBatch->actions = actions.batchBlock(batchID, batchSize);
    miniBatch->actionNoises = actionNoises.batchBlock(batchID, batchSize);
    miniBatch->costs = costs.block(0, batchID, maxLen, batchSize);

    if (useValue) miniBatch->values = values.block(0, batchID, maxLen, batchSize);
    if (useAdvantage) miniBatch->advantages = advantages.block(0, batchID, maxLen, batchSize);
//    miniBatch->stdevs = stdevs.block(0, batchID, ActionDim, batchSize);

    if (isRecurrent)  miniBatch->lengths = lengths.block(batchID, batchSize);
    miniBatch->termtypes = termtypes.block(batchID, batchSize);

    for (int i = 0; i < tensor3Ds.size(); i++) {
      miniBatch->tensor3Ds[i] = tensor3Ds[i].batchBlock(batchID, batchSize);
    }
    for (int i = 0; i < tensor2Ds.size(); i++) {
      miniBatch->tensor2Ds[i] = tensor2Ds[i].block(0, batchID, miniBatch->tensor2Ds[i].rows(), batchSize);
    }
    for (int i = 0; i < tensor1Ds.size(); i++) {
      miniBatch->tensor1Ds[i] = tensor1Ds[i].block(batchID, batchSize);
    }
  }

 public:
  ///method for additional data
  void append(Tensor1D &newData){
    tensor1Ds.push_back(newData);
    if(miniBatch) miniBatch->tensor1Ds.push_back(newData);
  }
  void append(Tensor2D &newData){
    tensor2Ds.push_back(newData);
    if(miniBatch) miniBatch->tensor2Ds.push_back(newData);
  }
  void append(Tensor3D &newData){
    tensor3Ds.push_back(newData);
    if(miniBatch) miniBatch->tensor3Ds.push_back(newData);
  }

  bool iterateBatch(int batchSize) {
    int cur_batch_size = batchSize;
    if (cur_batch_size >= batchNum - batchID || cur_batch_size == 0) {
      cur_batch_size = batchNum - batchID;
    }
    if (batchID >= batchNum) {
      batchID = 0;
      return false;
    }
    fillminiBatch(cur_batch_size);
    batchID += cur_batch_size;
    return true;
  }

  virtual void resize(int maxlen, int batches) {
    ///Keep first dimension.

    maxLen = maxlen;
    batchNum = batches;

    states.resize(StateDim, maxlen, batches);
    actions.resize(ActionDim, maxlen, batches);
    actionNoises.resize(ActionDim, maxlen, batches);

    costs.resize(maxlen, batches);
    if (useValue) values.resize(maxlen, batches);
    if (useAdvantage) advantages.resize(maxlen, batches);
//    stdevs.resize(ActionDim, batches);

    if (isRecurrent) lengths.resize(batches);
    termtypes.resize(batches);

    for (int i = 0; i < tensor3Ds.size(); i++) {
      miniBatch->tensor3Ds[i].resize(miniBatch->tensor3Ds[i].rows(),maxLen,batchNum);
    }
    for (int i = 0; i < tensor2Ds.size(); i++) {
      miniBatch->tensor2Ds[i].resize(miniBatch->tensor2Ds[i].rows(), batchNum);
    }
    for (int i = 0; i < tensor1Ds.size(); i++) {
      miniBatch->tensor1Ds[i].resize(batchNum);
    }
  }

  virtual void setZero() {
    states.setZero();
    actions.setZero();
    actionNoises.setZero();

    costs.setZero();
    if (useValue) values.setZero();
    if (useAdvantage) advantages.setZero();
//    stdevs.setZero();
    if (isRecurrent) lengths.setZero();
    termtypes.setZero();

    for (auto &ten3D: tensor3Ds) ten3D.setZero();
    for (auto &ten2D: tensor2Ds) ten2D.setZero();
    for (auto &ten1D: tensor1Ds) ten1D.setZero();
  }

  int maxLen;
  int batchNum;
  int batchID;
  bool useValue = false;
  bool useAdvantage = false;
  bool isRecurrent = false;

  Tensor3D states;
  Tensor3D actions;
  Tensor3D actionNoises;

  Tensor2D costs;
  Tensor2D values;
  Tensor2D advantages;
//  Tensor2D stdevs;

  Tensor1D lengths;
  Tensor1D termtypes;

  //vectors for additional data
  std::vector<rai::Tensor<Dtype, 3>> tensor3Ds;
  std::vector<rai::Tensor<Dtype, 2>> tensor2Ds;
  std::vector<rai::Tensor<Dtype, 1>> tensor1Ds;

  LearningData<Dtype, StateDim, ActionDim> *miniBatch;
};
}
}

#endif //RAI_LearningData_HPP
