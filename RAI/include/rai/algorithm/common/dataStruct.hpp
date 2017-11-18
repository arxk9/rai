//
// Created by joonho on 11/18/17.
//

#ifndef RAI_DATASTRUCT_HPP
#define RAI_DATASTRUCT_HPP

#include <rai/RAI_Tensor.hpp>

namespace rai {
namespace Algorithm {

template<typename Dtype, int stateDim, int actionDim>
struct history {
  history() : lengths("length"),
              states("state"),
              actions("sampled_oa"),
              actionNoises("noise_oa"), maxLen(0), batchNum(0), batchID(0) {
  };

  history(int maxlen, int batchNum_in) : lengths("length"),
                                         states("state"),
                                         actions("sampled_oa"),
                                         actionNoises("noise_oa"),
                                         maxLen(maxlen),
                                         batchNum(batchNum_in),
                                         batchID(0) { resize(maxlen, batchNum_in); };
  int maxLen;
  int batchNum;
  int batchID;

  /// history
  rai::Tensor<Dtype, 3> states;
  rai::Tensor<Dtype, 3> actions;
  rai::Tensor<Dtype, 2> costs;
  rai::Tensor<Dtype, 1> lengths;
  rai::Tensor<Dtype, 1> termtypes;

  /// distribution
  rai::Tensor<Dtype, 3> actionNoises;
  rai::Tensor<Dtype, 2> stdevs;

  history *minibatch;

  ///
  void setZero() {
    states.setZero();
    actions.setZero();
    costs.setZero();
    lengths.setZero();
    termtypes.setZero();
    actionNoises.setZero();
    stdevs.setZero();
  }

  void resize(int maxlen, int batchNum_in) {
    maxLen = maxlen;
    batchNum = batchNum_in;
    states.resize(stateDim, maxlen, batchNum_in);
    actions.resize(actionDim, maxlen, batchNum_in);
    costs.resize(maxlen, batchNum_in);
    lengths.resize(batchNum_in);
    termtypes.resize(batchNum_in);
    actionNoises.resize(actionDim, maxlen, batchNum_in);
    stdevs.resize(actionDim, batchNum_in);
  }

  bool iterateBatch(const int batchSize = 0, bool isrecurrent = false) {
    int cur_batch_size = batchSize;
    if (cur_batch_size >= batchNum - batchID || cur_batch_size == 0) {
      cur_batch_size = batchNum - batchID;
    }
    if (batchID >= batchNum) {
      batchID = 0;
      return false;
    }

    if (minibatch->batchNum != cur_batch_size || minibatch->maxLen != maxLen) {
      minibatch->batchNum = cur_batch_size;
      minibatch->maxLen = maxLen;

      minibatch->states.resize(stateDim, maxLen, cur_batch_size);
      minibatch->actions.resize(actionDim, maxLen, cur_batch_size);
      minibatch->costs.resize(maxLen, cur_batch_size);
      minibatch->termtypes.resize(cur_batch_size);
      minibatch->actionNoises.resize(actionDim, maxLen, cur_batch_size);
      minibatch->stdevs.resize(actionDim, cur_batch_size);
      if (isrecurrent) minibatch->lengths.resize(cur_batch_size);
    }

    minibatch->states = states.batchBlock(batchID, cur_batch_size);
    minibatch->actions = actions.batchBlock(batchID, cur_batch_size);
    minibatch->costs = costs.block(0, batchID, maxLen, cur_batch_size);
    minibatch->termtypes = termtypes.block(batchID, cur_batch_size);
    minibatch->actionNoises = actionNoises.batchBlock(batchID, cur_batch_size);
    minibatch->stdevs = stdevs.block(0, batchID, actionDim, cur_batch_size);

    if (isrecurrent) minibatch->lengths = lengths.block(batchID, cur_batch_size);

    batchID += cur_batch_size;
//    LOG(INFO) << "cur_ID/batchN = " << cur_ID << "/" << batchN << " batchsize = " << cur_batch_size;
    return true;
  };

};

template<typename Dtype, int stateDim, int actionDim>
struct historyWithAdvantage {
  historyWithAdvantage() : lengths("length"),
                           advantages("advantage"),
                           states("state"),
                           actions("sampled_oa"),
                           actionNoises("noise_oa"), maxLen(0), batchNum(0), batchID(0) {};

  historyWithAdvantage(int maxlen, int batchNum_in) : lengths("length"),
                                                      advantages("advantage"),
                                                      states("state"),
                                                      actions("sampled_oa"),
                                                      actionNoises("noise_oa"),
                                                      maxLen(maxlen),
                                                      batchNum(batchNum_in),
                                                      batchID(0) { resize(maxlen, batchNum_in); };
  int maxLen;
  int batchNum;
  int batchID;

  /// history
  rai::Tensor<Dtype, 3> states;
  rai::Tensor<Dtype, 3> actions;
  rai::Tensor<Dtype, 2> costs;
  rai::Tensor<Dtype, 1> lengths;
  rai::Tensor<Dtype, 1> termtypes;

  /// distribution
  rai::Tensor<Dtype, 3> actionNoises;
  rai::Tensor<Dtype, 2> stdevs;

  /// advantage
  rai::Tensor<Dtype, 2> advantages;

  historyWithAdvantage *minibatch;

  ///
  void setZero() {
    states.setZero();
    actions.setZero();
    costs.setZero();
    lengths.setZero();
    termtypes.setZero();
    actionNoises.setZero();
    stdevs.setZero();
    advantages.setZero();
  }

  void resize(int maxlen, int batchNum_in) {
    maxLen = maxlen;
    batchNum = batchNum_in;
    states.resize(stateDim, maxlen, batchNum_in);
    actions.resize(actionDim, maxlen, batchNum_in);
    costs.resize(maxlen, batchNum_in);
    lengths.resize(batchNum_in);
    termtypes.resize(batchNum_in);
    actionNoises.resize(actionDim, maxlen, batchNum_in);
    stdevs.resize(actionDim, batchNum_in);
    advantages.resize(maxlen,batchNum_in);
  }

  bool iterateBatch(const int batchSize = 0, bool isrecurrent = false) {
    int cur_batch_size = batchSize;
    if (cur_batch_size >= batchNum - batchID || cur_batch_size == 0) {
      cur_batch_size = batchNum - batchID;
    }
    if (batchID >= batchNum) {
      batchID = 0;
      return false;
    }
    if (minibatch->batchNum != cur_batch_size || minibatch->maxLen != maxLen) {
      minibatch->batchNum = cur_batch_size;
      minibatch->maxLen = maxLen;

      minibatch->states.resize(stateDim, maxLen, cur_batch_size);
      minibatch->actions.resize(actionDim, maxLen, cur_batch_size);
      minibatch->costs.resize(maxLen, cur_batch_size);
      minibatch->termtypes.resize(cur_batch_size);
      minibatch->actionNoises.resize(actionDim, maxLen, cur_batch_size);
      minibatch->stdevs.resize(actionDim, cur_batch_size);

      minibatch->advantages.resize(maxLen,cur_batch_size);
      if (isrecurrent) minibatch->lengths.resize(cur_batch_size);
    }

    minibatch->states = states.batchBlock(batchID, cur_batch_size);
    minibatch->actions = actions.batchBlock(batchID, cur_batch_size);
    minibatch->costs = costs.block(0, batchID, maxLen, cur_batch_size);
    minibatch->termtypes = termtypes.block(batchID, cur_batch_size);

    minibatch->actionNoises = actionNoises.batchBlock(batchID, cur_batch_size);
    minibatch->stdevs = stdevs.block(0, batchID, actionDim, cur_batch_size);

    minibatch->advantages = advantages.block(0, batchID, maxLen,cur_batch_size);
    if (isrecurrent) minibatch->lengths = lengths.block(batchID, cur_batch_size);

    batchID += cur_batch_size;
    return true;
  };
};

}
}

#endif //RAI_DATASTRUCT_HPP
