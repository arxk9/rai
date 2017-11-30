//
// Created by joonho on 11/21/17.
//

#ifndef RAI_RECURRENTDETERMINISTICPOLICY_HPP
#define RAI_RECURRENTDETERMINISTICPOLICY_HPP

#include "rai/function/common/DeterministicPolicy.hpp"
#include "rai/function/common/Qfunction.hpp"
#include "Qfunction_TensorFlow.hpp"
#include "common/ParameterizedFunction_TensorFlow.hpp"
#include "RecurrentQfunction_TensorFlow.hpp"

#pragma once

namespace rai {
namespace FuncApprox {

template<typename Dtype, int stateDim, int actionDim>
class RecurrentDeterministicPolicy_TensorFlow : public virtual DeterministicPolicy<Dtype, stateDim, actionDim>,
                                                public virtual ParameterizedFunction_TensorFlow<Dtype,
                                                                                                stateDim,
                                                                                                actionDim> {

 public:
  using PolicyBase = Policy<Dtype, stateDim, actionDim>;
  using Pfunction_tensorflow = ParameterizedFunction_TensorFlow<Dtype, stateDim, actionDim>;
  using Qfunction_tensorflow = RecurrentQfunction_TensorFlow<Dtype, stateDim, actionDim>;
  using Qfunction_ = Qfunction<Dtype, stateDim, actionDim>;
  typedef typename PolicyBase::State State;
  typedef typename PolicyBase::StateBatch StateBatch;
  typedef typename PolicyBase::Action Action;
  typedef typename PolicyBase::ActionBatch ActionBatch;
  typedef typename PolicyBase::Gradient Gradient;
  typedef typename PolicyBase::Jacobian Jacobian;
  typedef typename PolicyBase::JacobianWRTparam JacobianWRTparam;
  typedef typename PolicyBase::Jacobian JacobianWRTstate;
  typedef typename PolicyBase::JacoqWRTparam JacoqWRTparam;
  typedef typename PolicyBase::Tensor1D Tensor1D;
  typedef typename PolicyBase::Tensor2D Tensor2D;
  typedef typename PolicyBase::Tensor3D Tensor3D;
  typedef typename Pfunction_tensorflow ::HiddenState HiddenState;

  RecurrentDeterministicPolicy_TensorFlow(std::string pathToGraphDefProtobuf, Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(pathToGraphDefProtobuf, learningRate) {
  }

  RecurrentDeterministicPolicy_TensorFlow(std::string computeMode,
                                          std::string graphName,
                                          std::string graphParam,
                                          Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(
          "RecurrentDeterministicPolicy", computeMode, graphName, graphParam, learningRate), h("h_init") {
    hdim = this->getHiddenStatesize();
    h.resize(hdim, 0);
  }

  virtual void forward(State &state, Action &action) {
    std::vector<MatrixXD> vectorOfOutputs;
    MatrixXD h_, length;
    length.resize(1, 1);
    if (h.cols() != 1) {
      h_.resize(hdim, 1);
      h.setZero();
    }
    h_ = h.eMat();

    this->tf_->forward({{"state", state},
                        {"length", length},
                        {"h_init", h_}},
                       {"action"}, vectorOfOutputs);
    action = vectorOfOutputs[0];
    std::memcpy(action.data(), vectorOfOutputs[0].data(), sizeof(Dtype) * action.size());
    h.copyDataFrom(vectorOfOutputs[1]);
  }

  virtual void forward(StateBatch &states, ActionBatch &actions) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    Tensor3D stateT({stateDim, 1, states.cols()}, "state");
    Tensor1D len({states.cols()}, 1, "length");
    stateT.copyDataFrom(states);

    if (h.cols() != states.cols()) {
      h.resize(hdim, states.cols());
      h.setZero();
    }

    this->tf_->run({stateT, h, len}, {"action", "h_state"}, {}, vectorOfOutputs);
    std::memcpy(actions.data(), vectorOfOutputs[0].flat<Dtype>().data(), sizeof(Dtype) * actions.size());
    h.copyDataFrom(vectorOfOutputs[1]);
  }

  virtual void forward(Tensor3D &states, Tensor3D &actions) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    Tensor1D len({states.batches()}, states.dim(1), "length");

    if (h.cols() != states.batches()) {
      h.resize(hdim, states.batches());
      h.setZero();
    }

    this->tf_->run({states, h, len}, {"action", "h_state"}, {}, vectorOfOutputs);
    h.copyDataFrom(vectorOfOutputs[1]);
    actions.copyDataFrom(vectorOfOutputs[0]);
  }

  virtual Dtype performOneSolverIter(StateBatch &states, ActionBatch &actions) {
//    std::vector<MatrixXD> loss, dummy;
//    this->tf_->run({{"state", states},
//                    {"targetAction", actions},
//                    {"trainUsingTargetAction/learningRate", this->learningRate_}}, {"trainUsingTargetAction/loss"},
//                   {"trainUsingTargetAction/solver"}, loss);
//    this->tf_->run({{"state", states}}, {},
//                   {"action"}, dummy);
//    return loss[0](0);
    return 0;
  }

  virtual Dtype performOneSolverIter(history *minibatch, Tensor3D &actions) {
    std::vector<MatrixXD> vectorOfOutputs;
    actions = "targetAction";
    Tensor1D lr({1}, this->learningRate_(0), "trainUsingTargetQValue/learningRate");

    if (h.cols() != minibatch->batchNum) h.resize(hdim, minibatch->batchNum);
    h.setZero();

    this->tf_->run({minibatch->states, minibatch->lengths, actions, h, lr},
                   {"trainUsingTargetAction/loss"},
                   {"trainUsingTargetAction/solver"}, vectorOfOutputs);

    return vectorOfOutputs[0](0);
  };

  Dtype backwardUsingCritic(Qfunction_tensorflow *qFunction, history *minibatch) {
    std::vector<MatrixXD> dummy;
    Tensor3D gradients("trainUsingCritic/gradientFromCritic");
    Tensor1D lr({1}, this->learningRate_(0), "trainUsingCritic/learningRate");

    auto pQfunction = dynamic_cast<Qfunction_tensorflow const *>(qFunction);
    LOG_IF(FATAL, pQfunction == nullptr) << "You are mixing two different library types" << std::endl;
    gradients.resize(minibatch->actions.dim());
    if (h.cols() != minibatch->batchNum) h.resize(hdim, minibatch->batchNum);
    h.setZero();

    forward(minibatch->states, minibatch->actions);
    Dtype averageQ = pQfunction->getGradient_AvgOf_Q_wrt_action(minibatch, gradients);
//    std::cout << "grad" << std::endl << gradients<< std::endl;

    this->tf_->run({minibatch->states, minibatch->lengths, h, gradients, lr}, {"trainUsingCritic/gradnorm"},
                   {"trainUsingCritic/applyGradients"}, dummy);
    LOG(INFO) << dummy[0](0);
    return averageQ;
  }
  Dtype backwardUsingCritic(Qfunction_ *qFunction, StateBatch &states) {};

  Dtype getGradQwrtParam(Qfunction_ *qFunction, StateBatch &states, JacoqWRTparam &jaco) {
    return 0;
  }

  void getJacobianAction_WRT_LP(State &state, JacobianWRTparam &jacobian) {
  }

  virtual void getJacobianAction_WRT_State(State &state, JacobianWRTstate &jacobian) {
  }
  virtual bool isRecurrent() {
    return true;
  }

  virtual void reset(int n) {
    //n:index
    if (n >= h.cols())
      h.conservativeResize(hdim, n + 1);

    h.col(n).setZero();
  }

  virtual void terminate(int n) {
    int coldim = h.cols() - 1;
    LOG_IF(FATAL, coldim < 0) << "Initialize Innerstates first (Call reset)";
    LOG_IF(FATAL, n > coldim) << "n exceeds batchsize" << n << "vs." << coldim;
    h.removeCol(n);
  }

  virtual int getHiddenStatesize() {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    this->tf_->run({}, {"h_dim"}, {}, vectorOfOutputs);
    return vectorOfOutputs[0].scalar<int>()();
  }

  virtual void getHiddenStates(HiddenState &h_out){
    h_out = h;
  }

  int hiddenStateDim() { return hdim; }

 protected:
  using MatrixXD = typename TensorFlowNeuralNetwork<Dtype>::MatrixXD;
  int hdim = 0;
  Tensor2D h;

};
} // namespace FuncApprox
} // namespace rai


#endif //RAI_RECURRENTDETERMINISTICPOLICY_HPP
