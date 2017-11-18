//
// Created by joonho on 11/18/17.
//

#ifndef RAI_DATASTRUCT_HPP
#define RAI_DATASTRUCT_HPP

#include <rai/RAI_Tensor.hpp>


namespace rai{
namespace Algorithm {

template<typename Dtype, int stateDim, int actionDim>
struct history {
  history() : lengths("length"),
        states("state"),
        actions("sampled_oa"),
        actionNoises("noise_oa"),batchNum(0){
  };

  history(int maxlen, int batchNum_in) : lengths("length"),
              states("state"),
              actions("sampled_oa"),
              actionNoises("noise_oa"),batchNum(0),batchID(0){ resize(maxlen,batchNum_in); };

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

  history * minibatch;

  ///
  void resize(int maxlen, int batchNum_in){
    batchNum = batchNum_in;
    states.resize(stateDim,maxlen,batchNum_in);
    actions.resize(actionDim,maxlen,batchNum_in);
    costs.resize(maxlen,batchNum_in);
    lengths.resize(batchNum_in);
    termtypes.resize(batchNum_in);
    actionNoises.resize(actionDim,maxlen,batchNum_in);
    stdevs.resize(actionDim,batchNum_in);
  }


  bool iterateBatch(const int batchSize = 0, bool isrecurrent = false){
    int cur_batch_size = batchSize;
    if (cur_batch_size >=  batchNum - cur_ID || cur_batch_size == 0) {
      cur_batch_size =  batchN - cur_ID;
    }
    if(cur_ID >= batchN){
      cur_ID = 0;
      return false;
    }

    cur_minibatch.batchNum = cur_batch_size;
    cur_minibatch.states.resize(stateTensor.dim(0),stateTensor.dim(1),cur_batch_size);
    cur_minibatch.actions.resize(actionTensor.dim(0),actionTensor.dim(1),cur_batch_size);
    cur_minibatch.actionNoises.resize(actionNoiseTensor.dim(0),actionNoiseTensor.dim(1),cur_batch_size);
    cur_minibatch.advantages.resize(advantageTensor.dim(0),cur_batch_size);
    if(isrecurrent) cur_minibatch.lengths.resize(cur_batch_size);

    cur_minibatch.states = stateTensor.batchBlock(cur_ID, cur_batch_size);
    cur_minibatch.actions = actionTensor.batchBlock(cur_ID,cur_batch_size);
    cur_minibatch.actionNoises = actionNoiseTensor.batchBlock(cur_ID,cur_batch_size);
    cur_minibatch.advantages = advantageTensor.block(0, cur_ID, advantageTensor.dim(0),cur_batch_size);
    if(isrecurrent) cur_minibatch.lengths = trajLength.block(cur_ID,cur_batch_size);

    cur_ID +=cur_batch_size;
//    LOG(INFO) << "cur_ID/batchN = " << cur_ID << "/" << batchN << " batchsize = " << cur_batch_size;
    return true;
  };

};

template<typename Dtype>
struct historyWithAdvantage {
  historyWithAdvantage() : lengths("length"),
                           advantages("advantage"),
                           states("state"),
                           actions("sampled_oa"),
                           actionNoises("noise_oa"),batchNum(0){};

  int batchNum;

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
};


}
}

#endif //RAI_DATASTRUCT_HPP
