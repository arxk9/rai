//
// Created by joonho on 11/18/17.
//

#ifndef RAI_DATASTRUCT_HPP
#define RAI_DATASTRUCT_HPP

#include <rai/RAI_Tensor.hpp>

namespace rai {
namespace Algorithm {

///
template<typename Dtype>
struct TensorBatch {

  TensorBatch() : tensor1Ds(0), tensor2Ds(0), tensor3Ds(0), maxLen(0), batchNum(0), batchID(0) {
  };

  TensorBatch(int num1, int num2, int num3, int maxlen = 0, int batchNum_in = 0, bool isrecurrent = false)
      : tensor1Ds(num1),
        tensor2Ds(num2),
        tensor3Ds(num3),
        maxLen(maxlen),
        batchNum(batchNum_in),
        batchID(0),
        isrecurrent_(isrecurrent) {
  };

  int maxLen;
  int batchNum;
  int batchID;
  bool isrecurrent_;

  std::vector<rai::Tensor<Dtype, 3>> tensor3Ds;
  std::vector<rai::Tensor<Dtype, 2>> tensor2Ds;
  std::vector<rai::Tensor<Dtype, 1>> tensor1Ds;

//  TensorBatch *minibatch;

  virtual void setZero() {
    for (auto &ten3D: tensor3Ds) ten3D.setZero();
    for (auto &ten2D: tensor3Ds) ten2D.setZero();
    for (auto &ten1D: tensor3Ds) ten1D.setZero();
  }

  virtual void resize(int maxlen, int batchNum_in) {
    maxLen = maxlen;
    batchNum = batchNum_in;
  }

  virtual void partiallyfillBatch(int batchSize) = 0;
  virtual bool iterateBatch(int batchSize = 0) = 0;

};

template<typename Dtype, int stateDim, int actionDim>
struct history : public TensorBatch<Dtype> {
  typedef TensorBatch<Dtype> TensorBatch_;

  using TensorBatch_::TensorBatch;
  using TensorBatch_::batchNum;
  using TensorBatch_::batchID;
  using TensorBatch_::maxLen;
  using TensorBatch_::isrecurrent_;

  using TensorBatch_::tensor3Ds;
  using TensorBatch_::tensor2Ds;
  using TensorBatch_::tensor1Ds;
  typedef rai::Tensor<Dtype, 1> Tensor1D;
  typedef rai::Tensor<Dtype, 2> Tensor2D;
  typedef rai::Tensor<Dtype, 3> Tensor3D;

  history() : TensorBatch_::TensorBatch(2, 2, 3, 0, 0), minibatch(nullptr) {
    tensor3Ds[0] = "state";
    tensor3Ds[1] = "sampled_oa";
    tensor3Ds[2] = "noise_oa";
    tensor1Ds[0] = "length";
  };
  history(int maxlen, int batchNum_in, bool isrecurrent = false) :
      TensorBatch_::TensorBatch(2, 2, 3, maxlen, batchNum_in, isrecurrent),
      minibatch(nullptr) {
    tensor3Ds[0] = "state";
    tensor3Ds[1] = "sampled_oa";
    tensor3Ds[2] = "noise_oa";
    tensor1Ds[0] = "length";
  };

  /// name data
  Tensor3D &states = tensor3Ds[0];
  Tensor3D &actions = tensor3Ds[1];
  Tensor3D &actionNoises = tensor3Ds[2];
  Tensor2D &costs = tensor2Ds[0];
  Tensor2D &stdevs = tensor2Ds[1];
  Tensor1D &lengths = tensor1Ds[0];
  Tensor1D &termtypes = tensor1Ds[1];

//  rai::Tensor<Dtype, 3> states;
//  rai::Tensor<Dtype, 3> actions;
//  rai::Tensor<Dtype, 3> actionNoises;

//  rai::Tensor<Dtype, 1> lengths;
//  rai::Tensor<Dtype, 1> termtypes;

//  rai::Tensor<Dtype, 2> costs;
//  rai::Tensor<Dtype, 2> stdevs;

  history *minibatch;

  ///
  virtual void resize(int maxlen, int batchNum_in) {
    TensorBatch_::resize(maxlen, batchNum_in);
    states.resize(stateDim, maxlen, batchNum_in);
    actions.resize(actionDim, maxlen, batchNum_in);
    costs.resize(maxlen, batchNum_in);
    lengths.resize(batchNum_in);
    termtypes.resize(batchNum_in);
    actionNoises.resize(actionDim, maxlen, batchNum_in);
    stdevs.resize(actionDim, batchNum_in);
  }
  virtual void partiallyfillBatch(int batchSize) {
    minibatch->states = states.batchBlock(batchID, batchSize);
    minibatch->actions = actions.batchBlock(batchID, batchSize);
    minibatch->costs = costs.block(0, batchID, maxLen, batchSize);
    minibatch->termtypes = termtypes.block(batchID, batchSize);
    minibatch->actionNoises = actionNoises.batchBlock(batchID, batchSize);
    minibatch->stdevs = stdevs.block(0, batchID, actionDim, batchSize);
    if (isrecurrent_) minibatch->lengths = lengths.block(batchID, batchSize);
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

    if (minibatch->batchNum != cur_batch_size || minibatch->maxLen != maxLen) {
      minibatch->resize(maxLen, cur_batch_size);
    }
    partiallyfillBatch(cur_batch_size);
    batchID += cur_batch_size;
    return true;
  };
};

template<typename Dtype, int stateDim, int actionDim>
struct historyWithAdvantage : public TensorBatch<Dtype> {

  typedef TensorBatch<Dtype> TensorBatch_;
  using TensorBatch_::TensorBatch;
  using TensorBatch_::batchNum;
  using TensorBatch_::batchID;
  using TensorBatch_::maxLen;
  using TensorBatch_::isrecurrent_;

  using TensorBatch_::tensor3Ds;
  using TensorBatch_::tensor2Ds;
  using TensorBatch_::tensor1Ds;
  typedef rai::Tensor<Dtype, 1> Tensor1D;
  typedef rai::Tensor<Dtype, 2> Tensor2D;
  typedef rai::Tensor<Dtype, 3> Tensor3D;

  historyWithAdvantage() : TensorBatch_::TensorBatch(2, 3, 3, 0, 0), minibatch(nullptr) {
    tensor3Ds[0] = "state";
    tensor3Ds[1] = "sampled_oa";
    tensor3Ds[2] = "noise_oa";
    tensor2Ds[2] = "advantage";
    tensor1Ds[0] = "length";
  };
  historyWithAdvantage(int maxlen, int batchNum_in, bool isrecurrent = false) :
      TensorBatch_::TensorBatch(2, 3, 3, maxlen, batchNum_in, isrecurrent),
      minibatch(nullptr) {
    tensor3Ds[0] = "state";
    tensor3Ds[1] = "sampled_oa";
    tensor3Ds[2] = "noise_oa";
    tensor2Ds[2] = "advantage";
    tensor1Ds[0] = "length";
  };

  /// name data
  Tensor3D &states = tensor3Ds[0];
  Tensor3D &actions = tensor3Ds[1];
  Tensor3D &actionNoises = tensor3Ds[2];

  Tensor2D &costs = tensor2Ds[0];
  Tensor2D &stdevs = tensor2Ds[1];
  Tensor2D &advantages = tensor2Ds[2];

  Tensor1D &lengths = tensor1Ds[0];
  Tensor1D &termtypes = tensor1Ds[1];

  historyWithAdvantage *minibatch;

  virtual void resize(int maxlen, int batchNum_in) {
    TensorBatch_::resize(maxlen, batchNum_in);
    states.resize(stateDim, maxlen, batchNum_in);
    actions.resize(actionDim, maxlen, batchNum_in);
    costs.resize(maxlen, batchNum_in);
    lengths.resize(batchNum_in);
    termtypes.resize(batchNum_in);
    actionNoises.resize(actionDim, maxlen, batchNum_in);
    stdevs.resize(actionDim, batchNum_in);
    advantages.resize(maxlen, batchNum_in);

  }
  virtual void partiallyfillBatch(int batchSize) {
    minibatch->states = states.batchBlock(batchID, batchSize);
    minibatch->actions = actions.batchBlock(batchID, batchSize);
    minibatch->costs = costs.block(0, batchID, maxLen, batchSize);
    minibatch->termtypes = termtypes.block(batchID, batchSize);
    minibatch->actionNoises = actionNoises.batchBlock(batchID, batchSize);
    minibatch->stdevs = stdevs.block(0, batchID, actionDim, batchSize);
    if (isrecurrent_) minibatch->lengths = lengths.block(batchID, batchSize);
    minibatch->advantages = advantages.block(0, batchID, maxLen, batchSize);
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
    if (minibatch->batchNum != cur_batch_size || minibatch->maxLen != maxLen) {
      minibatch->resize(maxLen, cur_batch_size);
    }
    partiallyfillBatch(cur_batch_size);
    batchID += cur_batch_size;
    return true;
  };
};

}
}

#endif //RAI_DATASTRUCT_HPP
