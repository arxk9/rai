  #include "rai/function/common/Qfunction.hpp"
#include "common/RecurrentParametrizedFunction_TensorFlow.hpp"

#pragma once

namespace rai {
namespace FuncApprox {

template<typename Dtype, int stateDim, int actionDim>
class RecurrentQfunction_TensorFlow : public virtual RecurrentParameterizedFunction_TensorFlow<Dtype,
                                                                                      stateDim + actionDim, 1>,
                                      public virtual Qfunction<Dtype, stateDim, actionDim> {

 public:
  typedef Eigen::Matrix<Dtype, -1, 1> VectorXD;
  typedef Eigen::Matrix<Dtype, -1, -1> MatrixXD;
  using QfunctionBase = Qfunction<Dtype, stateDim, actionDim>;
  using Pfunction_tensorflow = RecurrentParameterizedFunction_TensorFlow<Dtype, stateDim + actionDim, 1>;

  using Pfunction_tensorflow::h;
  using Pfunction_tensorflow::hdim;

  typedef typename QfunctionBase::State State;
  typedef typename QfunctionBase::StateBatch StateBatch;

  typedef typename QfunctionBase::Action Action;
  typedef typename QfunctionBase::ActionBatch ActionBatch;
  typedef typename QfunctionBase::Jacobian Jacobian;
  typedef typename QfunctionBase::Value Value;
  typedef typename QfunctionBase::ValueBatch ValueBatch;
  typedef typename QfunctionBase::Tensor1D Tensor1D;
  typedef typename QfunctionBase::Tensor2D Tensor2D;
  typedef typename QfunctionBase::Tensor3D Tensor3D;
  typedef typename QfunctionBase::Dataset Dataset;

  RecurrentQfunction_TensorFlow(std::string pathToGraphDefProtobuf, Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::RecurrentParameterizedFunction_TensorFlow(pathToGraphDefProtobuf, learningRate) {
  }

  RecurrentQfunction_TensorFlow(std::string computeMode,
                                std::string graphName,
                                std::string graphParam,
                                Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::RecurrentParameterizedFunction_TensorFlow(
          "RecurrentQfunction", computeMode, graphName, graphParam, learningRate) {
  }

  virtual void forward(State &state, Action &action, Dtype &value) {
    std::vector<MatrixXD> vectorOfOutputs;
    MatrixXD h_, length;
    length.resize(1,1);
    h_.resize(hdim,1);
    h_.setZero();

    this->tf_->run({{"state", state},
                    {"sampledAction", action},
                    {"length", length},
                    {"h_init", h_}},
                   {"QValue"}, {}, vectorOfOutputs);
    value = vectorOfOutputs[0](0);
  }

  virtual void forward(StateBatch &states, ActionBatch &actions, ValueBatch &values) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    Tensor3D stateT({stateDim, 1, states.cols()}, "state");
    Tensor3D actionT({actionDim, 1, states.cols()}, "sampledAction");
    Tensor1D len({states.cols()}, 1, "length");
    stateT.copyDataFrom(states);

    if (h.cols() != states.cols()) {
      h.resize(hdim, states.cols());
    }
      h.setZero();

    this->tf_->run({stateT, actionT, h, len}, {"QValue", "h_state"}, {}, vectorOfOutputs);
    std::memcpy(values.data(), vectorOfOutputs[0].flat<Dtype>().data(), sizeof(Dtype) * values.size());
    h.copyDataFrom(vectorOfOutputs[1]);
  }

  virtual void test(Tensor3D &states, Tensor3D &actions) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    Tensor1D len({states.batches()}, states.dim(1), "length");

    if (h.cols() != states.batches()) {
      h.resize(hdim, states.batches());
    }
      h.setZero();
//    }

    Eigen::Matrix<Dtype,-1,1> test;
    this->tf_->run({states, actions, h, len}, {"test"}, {}, vectorOfOutputs);
    test.resize(vectorOfOutputs[0].template flat<Dtype>().size());
    std::memcpy(test.data(), vectorOfOutputs[0].template flat<Dtype>().data(), sizeof(Dtype) * test.size());
    LOG(INFO) << test.transpose();
  }


  virtual Dtype performOneSolverIter(StateBatch& states, ActionBatch& actions, ValueBatch &values){
  LOG(FATAL) << "NOT IMPLEMENTED";
//    std::vector<MatrixXD> vectorOfOutputs;
//    this->tf_->run({{"state", states},
//                    {"sampledAction", actions},
//                    {"targetQValue", values},
//                    {"trainUsingTargetQValue/learningRate", this->learningRate_}},
//                   {"trainUsingTargetQValue/loss"},
//                   {"trainUsingTargetQValue/solver"}, vectorOfOutputs);
//    return vectorOfOutputs[0](0);
  return 0;
  };

  virtual Dtype performOneSolverIter( Tensor3D &states,  Tensor3D &actions, Tensor1D &lengths,Tensor3D &values){
    std::vector<MatrixXD> vectorOfOutputs;
    values = "targetQValue";
    Tensor1D lr({1}, this->learningRate_(0), "trainUsingTargetQValue/learningRate");

    if(h.cols()!= states.batches()) h.resize(hdim, states.batches());
    h.setZero();

    this->tf_->run({states, actions, lengths, values, h, lr},
                   {"trainUsingTargetQValue/loss"},
                   {"trainUsingTargetQValue/solver"}, vectorOfOutputs);

    return vectorOfOutputs[0](0);
  };

  virtual Dtype performOneSolverIter(Dataset *minibatch, Tensor3D &values){
    std::vector<MatrixXD> vectorOfOutputs;
    values = "targetQValue";
    Tensor1D lr({1}, this->learningRate_(0), "trainUsingTargetQValue/learningRate");

    if(h.cols()!= minibatch->batchNum) h.resize(hdim, minibatch->batchNum);
    h.setZero();

    this->tf_->run({minibatch->states, minibatch->actions, minibatch->lengths, values, h, lr},
                    {"trainUsingTargetQValue/loss"},
                   {"trainUsingTargetQValue/solver"}, vectorOfOutputs);

    return vectorOfOutputs[0](0);
  };


  virtual Dtype test(Dataset *minibatch, Tensor3D &values){
    std::vector<MatrixXD> vectorOfOutputs;
    Tensor1D lr({1}, this->learningRate_(0), "trainUsingTargetQValue/learningRate");
    values = "targetQValue";
    if(h.cols()!= minibatch->batchNum) h.resize(hdim, minibatch->batchNum);
    h.setZero();

    Eigen::Matrix<Dtype,-1,1> test;
    this->tf_->run({minibatch->states, minibatch->actions, minibatch->lengths, values, h, lr},
                   {"test"},
                   {}, vectorOfOutputs);

    test.resize(vectorOfOutputs[0].size());
    std::memcpy(test.data(), vectorOfOutputs[0].data(), sizeof(Dtype) * test.size());
    LOG(INFO) << test.transpose();
    return vectorOfOutputs[0](0);
  };

  Dtype getGradient_AvgOf_Q_wrt_action(Dataset *minibatch, Tensor3D &gradients) const
  {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    Tensor2D hiddenState({hdim, minibatch->batchNum}, "h_init");
    hiddenState = minibatch->hiddenStates.col(0);

    this->tf_->run({minibatch->states,
                    minibatch->actions, minibatch->lengths,hiddenState},
                   {"gradient_AvgOf_Q_wrt_action", "average_Q_value"}, {}, vectorOfOutputs);
    gradients = (vectorOfOutputs[0]);
    return vectorOfOutputs[1].scalar<Dtype>()();
  }
};
}
}