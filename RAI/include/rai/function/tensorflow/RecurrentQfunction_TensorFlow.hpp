#include "rai/function/common/Qfunction.hpp"
#include "common/ParameterizedFunction_TensorFlow.hpp"

#pragma once

namespace rai {
namespace FuncApprox {

template<typename Dtype, int stateDim, int actionDim>
class RecurrentQfunction_TensorFlow : public virtual ParameterizedFunction_TensorFlow<Dtype,
                                                                                      stateDim + actionDim, 1>,
                                      public virtual Qfunction<Dtype, stateDim, actionDim> {

 public:
  using QfunctionBase = Qfunction<Dtype, stateDim, actionDim>;
  using Pfunction_tensorflow = ParameterizedFunction_TensorFlow<Dtype, stateDim + actionDim, 1>;
  typedef typename QfunctionBase::State State;
  typedef typename QfunctionBase::StateBatch StateBatch;
  using RecurrentState = Eigen::VectorXd;
  using RecurrentStateBatch = Eigen::MatrixXd;
  typedef typename QfunctionBase::Action Action;
  typedef typename QfunctionBase::ActionBatch ActionBatch;
  typedef typename QfunctionBase::Jacobian Jacobian;
  typedef typename QfunctionBase::Value Value;
  typedef typename QfunctionBase::ValueBatch ValueBatch;
  typedef Eigen::Matrix<Dtype, actionDim, Eigen::Dynamic> JacobianQwrtActionBatch;

  RecurrentQfunction_TensorFlow(std::string pathToGraphDefProtobuf, Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(pathToGraphDefProtobuf, learningRate) {
  }

  RecurrentQfunction_TensorFlow(std::string computeMode,
                                std::string graphName,
                                std::string graphParam,
                                Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(
          "RecurrentQfunction", computeMode, graphName, graphParam, learningRate) {
  }

  virtual void forward(State &state, RecurrentState &rcrntState, Action &action, Dtype &value) {
    rai::Vector<MatrixXD> vectorOfOutputs;
    this->tf_->run({{"state", state},
                    {"init_rcrnt_state", rcrntState},
                    {"action", action},
                    {"updateBNparams", this->notUpdateBN}},
                   {"QValue"}, {}, vectorOfOutputs);
    value = vectorOfOutputs[0](0);
  }

  virtual void forward(StateBatch &states, RecurrentStateBatch &rcrntStates, ActionBatch &actions, ValueBatch &values) {
    rai::Vector<MatrixXD> vectorOfOutputs;
    this->tf_->run({{"state", states},
                    {"init_rcrnt_state", rcrntStates},
                    {"action", actions},
                    {"updateBNparams", this->notUpdateBN}},
                   {"QValue"}, {}, vectorOfOutputs);
    values = vectorOfOutputs[0];
  }

  Dtype performOneSolverIter(rai::Vector<StateBatch> &states,
                             rai::Vector<RecurrentStateBatch> &rcrntStates,
                             rai::Vector<ActionBatch> &actions,
                             rai::Vector<ValueBatch> &values) {
    rai::Vector<MatrixXD> outputs, dummy;
    this->tf_->run({{"state", states},
                    {"new_rcrnt_state", rcrntStates},
                    {"action", actions},
                    {"targetQValue", values},
                    {"trainUsingTargetQValue/learningRate", this->learningRate_},
                    {"updateBNparams",
                     this->notUpdateBN}}, {"trainUsingTargetQValue/loss"},
                   {"trainUsingTargetQValue/solver"}, outputs);
    this->tf_->run({{"state", states},
                    {"action", actions},
                    {"updateBNparams",
                     this->updateBN}}, {},
                   {"QValue"}, dummy);
    return outputs[0](0);
  }

  Dtype getGradient_AvgOf_Q_wrt_action(StateBatch &states, RecurrentStateBatch &rcrntStates, ActionBatch &actions,
                                       JacobianQwrtActionBatch &gradients) const {
    rai::Vector<MatrixXD> outputs;
    this->tf_->run({{"state", states},
                    {"action", actions},
                    {"updateBNparams", this->notUpdateBN}},
                   {"gradient_AvgOf_Q_wrt_action", "average_Q_value"}, {}, outputs);
    gradients = outputs[0];
    return outputs[1](0);
  }

 protected:
  using MatrixXD = typename TensorFlowNeuralNetwork<Dtype>::MatrixXD;

};
}
}