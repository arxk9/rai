//
// Created by joonho on 11/21/17.
//

#ifndef RAI_DATASTRUCT_HPP
#define RAI_DATASTRUCT_HPP


#include "rai/tasks/common/Task.hpp"
//#include "rai/function/common/ValueFunction.hpp"
#include <rai/common/VectorHelper.hpp>
#include <rai/RAI_core>
#include <rai/memory/Trajectory.hpp>

namespace rai {
namespace Algorithm {

template<typename Dtype>
struct TensorBatch {
  TensorBatch() : tensor1Ds(0), tensor2Ds(0), tensor3Ds(0), maxLen(0), batchNum(0), batchID(0),
                  isrecurrent_(false) {
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

  virtual void setZero() {
    for (auto &ten3D: tensor3Ds) ten3D.setZero();
    for (auto &ten2D: tensor3Ds) ten2D.setZero();
    for (auto &ten1D: tensor3Ds) ten1D.setZero();
  }

  virtual void resize(int maxlen, int batchNum_in) {
    maxLen = maxlen;
    batchNum = batchNum_in;
  }

  void fillminibatch(TensorBatch *minibatch, int batchSize = 0) {

    if (minibatch->batchNum != batchSize || minibatch->maxLen != maxLen) {
      minibatch->resize(maxLen, batchSize);
    }
    for (int i = 0; i < tensor3Ds.size(); i++) {
      minibatch->tensor3Ds[i] = tensor3Ds[i].batchBlock(batchID, batchSize);
    }
    for (int i = 0; i < tensor2Ds.size(); i++) {
      minibatch->tensor2Ds[i] = tensor2Ds[i].block(0, batchID, minibatch->tensor2Ds[i].rows(), batchSize);
    }
    for (int i = 0; i < tensor1Ds.size(); i++) {
      minibatch->tensor1Ds[i] = tensor1Ds[i].block(batchID, batchSize);

    }
  }

  bool iterateBatch(TensorBatch *minibatch, int batchSize) {
    int cur_batch_size = batchSize;
    if (cur_batch_size >= batchNum - batchID || cur_batch_size == 0) {
      cur_batch_size = batchNum - batchID;
    }
    if (batchID >= batchNum) {
      batchID = 0;
      return false;
    }
    fillminibatch(minibatch, cur_batch_size);
    batchID += cur_batch_size;
    return true;
  }
  virtual bool iterateBatch(int batchSize) = 0;

 public:
  int maxLen;
  int batchNum;
  int batchID;
  bool isrecurrent_;

  std::vector<rai::Tensor<Dtype, 3>> tensor3Ds;
  std::vector<rai::Tensor<Dtype, 2>> tensor2Ds;
  std::vector<rai::Tensor<Dtype, 1>> tensor1Ds;
};

template<typename Dtype, int stateDim, int actionDim>
struct history : public TensorBatch<Dtype> {
  typedef TensorBatch<Dtype> TensorBatch_;
  typedef rai::Tensor<Dtype, 1> Tensor1D;
  typedef rai::Tensor<Dtype, 2> Tensor2D;
  typedef rai::Tensor<Dtype, 3> Tensor3D;

  using TensorBatch_::batchNum;
  using TensorBatch_::batchID;
  using TensorBatch_::maxLen;
  using TensorBatch_::isrecurrent_;

  using TensorBatch_::tensor3Ds;
  using TensorBatch_::tensor2Ds;
  using TensorBatch_::tensor1Ds;

  history() : TensorBatch_::TensorBatch(2, 2, 3, 0, 0), minibatch(nullptr) {
    tensor3Ds[0] = "state";
    tensor3Ds[1] = "sampledAction";
    tensor3Ds[2] = "actionNoise";
    tensor2Ds[0] = "costs";
    tensor2Ds[1] = "stdevs";
    tensor1Ds[0] = "length";
    tensor1Ds[1] = "termtypes";
  };
  history(int maxlen, int batchNum_in, bool isrecurrent = false) :
      TensorBatch_::TensorBatch(2, 2, 3, maxlen, batchNum_in, isrecurrent), minibatch(nullptr) {
    tensor3Ds[0] = "state";
    tensor3Ds[1] = "sampledAction";
    tensor3Ds[2] = "actionNoise";
    tensor2Ds[0] = "costs";
    tensor2Ds[1] = "stdevs";
    tensor1Ds[0] = "length";
    tensor1Ds[1] = "termtypes";
  };

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

  virtual bool iterateBatch(int batchSize) {
    return TensorBatch_::iterateBatch(minibatch, batchSize);
  };

  /// assign data
  Tensor3D &states = tensor3Ds[0];
  Tensor3D &actions = tensor3Ds[1];
  Tensor3D &actionNoises = tensor3Ds[2];
  Tensor2D &costs = tensor2Ds[0];
  Tensor2D &stdevs = tensor2Ds[1];
  Tensor1D &lengths = tensor1Ds[0];
  Tensor1D &termtypes = tensor1Ds[1];

  history *minibatch;
};

template<typename Dtype, int stateDim, int actionDim>
struct historyWithAdvantage : public TensorBatch<Dtype> {

  typedef TensorBatch<Dtype> TensorBatch_;
  typedef rai::Tensor<Dtype, 1> Tensor1D;
  typedef rai::Tensor<Dtype, 2> Tensor2D;
  typedef rai::Tensor<Dtype, 3> Tensor3D;

  using TensorBatch_::batchNum;
  using TensorBatch_::batchID;
  using TensorBatch_::maxLen;
  using TensorBatch_::isrecurrent_;

  using TensorBatch_::tensor3Ds;
  using TensorBatch_::tensor2Ds;
  using TensorBatch_::tensor1Ds;
  using ValueFunc_ = rai::FuncApprox::ValueFunction<Dtype, stateDim>;
  using Task_ = rai::Task::Task<Dtype, stateDim, actionDim, 0>;

  historyWithAdvantage() : TensorBatch_::TensorBatch(2, 3, 3, 0, 0), minibatch(nullptr) {
    tensor3Ds[0] = "state";
    tensor3Ds[1] = "sampledAction";
    tensor3Ds[2] = "actionNoise";
    tensor2Ds[0] = "costs";
    tensor2Ds[1] = "stdevs";
    tensor2Ds[2] = "advantage";
    tensor1Ds[0] = "length";
    tensor1Ds[1] = "termtypes";
  };
  historyWithAdvantage(int maxlen, int batchNum_in, bool isrecurrent = false) :
      TensorBatch_::TensorBatch(2, 3, 3, maxlen, batchNum_in, isrecurrent), minibatch(nullptr) {
    tensor3Ds[0] = "state";
    tensor3Ds[1] = "sampledAction";
    tensor3Ds[2] = "actionNoise";
    tensor2Ds[0] = "costs";
    tensor2Ds[1] = "stdevs";
    tensor2Ds[2] = "advantage";
    tensor1Ds[0] = "length";
    tensor1Ds[1] = "termtypes";
  };

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

  virtual bool iterateBatch(int batchSize) {
    return TensorBatch_::iterateBatch(minibatch, batchSize);
  };
  /// assign data
  Tensor3D &states = tensor3Ds[0];
  Tensor3D &actions = tensor3Ds[1];
  Tensor3D &actionNoises = tensor3Ds[2];
  Tensor2D &costs = tensor2Ds[0];
  Tensor2D &stdevs = tensor2Ds[1];
  Tensor2D &advantages = tensor2Ds[2];
  Tensor1D &lengths = tensor1Ds[0];
  Tensor1D &termtypes = tensor1Ds[1];
  historyWithAdvantage *minibatch;
};

}
}

#endif //RAI_DATASTRUCT_HPP
