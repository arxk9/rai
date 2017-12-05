//
// Created by jhwangbo on 08/08/17.
// This class's old data is deleted when you acquire new data
//

#ifndef RAI_LearningData_HPP
#define RAI_LearningData_HPP

#include <rai/memory/Trajectory.hpp>
#include <rai/RAI_core>
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

 public:
  LearningData()
      : maxLen(0),
        batchNum(0),
        batchID(0),
        miniBatch(nullptr),
        extraTensor1D(0),
        extraTensor2D(0),
        extraTensor3D(0),
        hDim(-1) {
    states = "state";
    actions = "sampledAction";
    actionNoises = "actionNoise";
//    hiddenStates = "h_init";

    costs = "costs";
    values = "targetValue";
    advantages = "advantage";
//    stdevs = "stdevs";

    lengths = "length";
    termtypes = "termtypes";
  };

  ///method for additional data
  void append(Tensor1D &newData) {
    if (newData.size() == -1) newData.resize(0); //rai Tensor is not initialized
    extraTensor1D.push_back(newData);
    if (miniBatch) miniBatch->extraTensor1D.push_back(newData);
  }
  void append(Tensor2D &newData) {
    if (newData.size() == -1) newData.resize(0, 0); //rai Tensor is not initialized
    extraTensor2D.push_back(newData);
    if (miniBatch) miniBatch->extraTensor2D.push_back(newData);
  }
  void append(Tensor3D &newData) {
    if (newData.size() == -1) newData.resize(0, 0, 0); //rai Tensor is not initialized
    extraTensor3D.push_back(newData);
    if (miniBatch) miniBatch->extraTensor3D.push_back(newData);
  }

  bool iterateBatch(int batchSize) {
    int cur_batch_size = batchSize;
    if (cur_batch_size >= batchNum - batchID || cur_batch_size == 0) {
      cur_batch_size = batchNum - batchID;
    }
//    LOG(INFO) << batchID << "batchID ";

    if (batchID >= batchNum) {
      batchID = 0;
      return false;
    }
    fillminiBatch(cur_batch_size);
    batchID += cur_batch_size;
    return true;
  }

  void divideSequences(int segLen, int stride = 1, bool keepHiddenState = true) {
    ///Implementation of truncated BPTT
    ///k1: stride
    ///k2: len

    LOG_IF(FATAL, segLen > maxLen) << "segLen should be smaller than this->maxLen";

    ///copyData
    Tensor3D states_t(states);
    Tensor3D actions_t(actions);
    Tensor3D actionNoises_t(actionNoises);
    Tensor3D hiddenStates_t(hiddenStates);
//  Tensor2D costs;
//  Tensor2D stdevs;

    Tensor2D values_t(values);
    Tensor2D advantages_t(advantages);
    Tensor1D lengths_t(lengths);
    Tensor1D termtypes_t(termtypes);

    int batchNum_t = batchNum;
    //vectors for additional data
    std::vector<rai::Tensor<Dtype, 3>> extraTensor3D_t;
    std::vector<rai::Tensor<Dtype, 2>> extraTensor2D_t;
    std::vector<rai::Tensor<Dtype, 1>> extraTensor1D_t;
    for (int i = 0; i < extraTensor1D.size(); i++)
      extraTensor1D_t.push_back(extraTensor1D[i]);
    for (int i = 0; i < extraTensor2D.size(); i++)
      extraTensor2D_t.push_back(extraTensor2D[i]);
    for (int i = 0; i < extraTensor3D.size(); i++)
      extraTensor3D_t.push_back(extraTensor3D[i]);

    int expandedBatchNum = 0;
    int segNum[batchNum];
    for (int i = 0; i < batchNum; i++) {
      segNum[i] = std::ceil((lengths[i] - segLen) / stride) + 1;
      expandedBatchNum += segNum[i];
    }

    resize(segLen, expandedBatchNum);

    hiddenStates.setZero();
    int segID = 0;
    int position = 0;
    for (int i = 0; i < batchNum_t; i++) {

      for (int j = 0; j < segNum[i]; j++) {
        position = stride * j;
        if (position > lengths_t[i] - segLen) position = std::max(0, (int)lengths_t[i] - segLen); //to make last segment full
        states.batch(segID) = states_t.batch(i).block(0, position, StateDim, segLen);
        actions.batch(segID) = actions_t.batch(i).block(0, position, ActionDim, segLen);
        actionNoises.batch(segID) = actionNoises_t.batch(i).block(0, position, ActionDim, segLen);

        if (useValue) values.col(segID) = values_t.block(position,i,segLen,1);
        if (useAdvantage) advantages.col(segID) = advantages_t.block(position, i, segLen, 1);

        lengths[segID] = std::min(segLen, (int)lengths_t[i]);
        termtypes[segID] = termtypes_t[i];

        for (int k = 0; k < extraTensor1D.size(); k++)
          extraTensor1D[k][segID] = extraTensor1D_t[k][i];
        for (int k = 0; k < extraTensor2D.size(); k++)
          extraTensor2D[k].col(segID) = extraTensor2D_t[k].block(position,i,segLen,1);
        for (int k = 0; k < extraTensor3D.size(); k++)
          extraTensor3D[k].batch(segID) = extraTensor3D_t[k].batch(i).block(0,position,extraTensor3D_t[k].dim(0),segLen);

        if (keepHiddenState) {
          /// We only use first column.
          hiddenStates.batch(segID).col(0) = hiddenStates_t.batch(i).col(position);
//          hiddenStates.batch(segID) =  hiddenStates_t.batch(segID).block(0,stride*j,hDim,segLen);
        }
        segID++;
      }
    }
//    std::cout << states_t << std::endl;
//    std::cout << states.batch(0) << std::endl;
//    std::cout << states.batch(1) << std::endl;
//    std::cout << states.batch(2) << std::endl;

  }

  void resize(int maxlen, int batches) {
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
    if (isRecurrent) hiddenStates.resize(hDim, maxlen, batches);
    termtypes.resize(batches);


    for (int k = 0; k < extraTensor1D.size(); k++)
      extraTensor1D[k].resize(batches);
    for (int k = 0; k < extraTensor2D.size(); k++)
      extraTensor2D[k].resize(maxlen,batches);
    for (int k = 0; k < extraTensor3D.size(); k++)
      extraTensor3D[k].resize(extraTensor3D[k].dim(0), maxlen, batches);

  }

  void setZero() {
    states.setZero();
    actions.setZero();
    actionNoises.setZero();

    costs.setZero();
    if (useValue) values.setZero();
    if (useAdvantage) advantages.setZero();
//    stdevs.setZero();
    if (isRecurrent) lengths.setZero();
    termtypes.setZero();

    for (auto &ten3D: extraTensor3D) ten3D.setZero();
    for (auto &ten2D: extraTensor2D) ten2D.setZero();
    for (auto &ten1D: extraTensor1D) ten1D.setZero();
  }

 private:
  void fillminiBatch(int batchSize = 0) {

    if (miniBatch->batchNum != batchSize || miniBatch->maxLen != maxLen) {
      miniBatch->resize(maxLen, batchSize);
    }

    miniBatch->states = states.batchBlock(batchID, batchSize);
    miniBatch->actions = actions.batchBlock(batchID, batchSize);
    miniBatch->actionNoises = actionNoises.batchBlock(batchID, batchSize);
//    miniBatch->costs = costs.block(0, batchID, maxLen, batchSize);
//    miniBatch->stdevs = stdevs.block(0, batchID, ActionDim, batchSize);

    if (useValue) miniBatch->values = values.block(0, batchID, maxLen, batchSize);
    if (useAdvantage) miniBatch->advantages = advantages.block(0, batchID, maxLen, batchSize);

    if (isRecurrent) miniBatch->lengths = lengths.block(batchID, batchSize);
    if (isRecurrent) miniBatch->hiddenStates = hiddenStates.batchBlock(batchID, batchSize);

    miniBatch->termtypes = termtypes.block(batchID, batchSize);

    for (int i = 0; i < extraTensor3D.size(); i++) {
      if (miniBatch->extraTensor3D[i].batches() != batchSize
          || miniBatch->extraTensor3D[i].rows() != extraTensor3D[i].rows()
          || miniBatch->extraTensor3D[i].cols() != extraTensor3D[i].cols())
        miniBatch->extraTensor3D[i].resize(extraTensor3D[i].rows(), extraTensor3D[i].cols(), batchSize);
      miniBatch->extraTensor3D[i] = extraTensor3D[i].batchBlock(batchID, batchSize);
    }
    for (int i = 0; i < extraTensor2D.size(); i++) {
      if (miniBatch->extraTensor2D[i].cols() != batchSize
          || miniBatch->extraTensor2D[i].rows() != extraTensor2D[i].rows())
        miniBatch->extraTensor2D[i].resize(extraTensor2D[i].rows(), batchSize);

      miniBatch->extraTensor2D[i] = extraTensor2D[i].block(0, batchID, extraTensor2D[i].rows(), batchSize);
    }
    for (int i = 0; i < extraTensor1D.size(); i++) {
      if (miniBatch->extraTensor1D[i].dim(0) != batchSize != batchSize)
        miniBatch->extraTensor1D[i].resize(batchSize);
      miniBatch->extraTensor1D[i] = extraTensor1D[i].block(batchID, batchSize);
    }
  }

 public:
  int maxLen;
  int batchNum;
  int batchID;
  int hDim;

  bool useValue = false;
  bool useAdvantage = false;
  bool isRecurrent = false;

  Tensor3D states;
  Tensor3D actions;
  Tensor3D actionNoises;
  Tensor3D hiddenStates;
  Tensor2D costs;
  Tensor2D values;
  Tensor2D advantages;
//  Tensor2D stdevs;

  Tensor1D lengths;
  Tensor1D termtypes;

  //vectors for additional data
  std::vector<rai::Tensor<Dtype, 3>> extraTensor3D;
  std::vector<rai::Tensor<Dtype, 2>> extraTensor2D;
  std::vector<rai::Tensor<Dtype, 1>> extraTensor1D;

  LearningData<Dtype, StateDim, ActionDim> *miniBatch;
};
}
}

#endif //RAI_LearningData_HPP
