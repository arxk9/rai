#include "rai/function/common/Qfunction.hpp"
#include "common/ParameterizedFunction_TensorFlow.hpp"

#pragma once

namespace RAI {
namespace FuncApprox {

template<typename Dtype, int stateDim, int actionDim>
class Qfunction_TensorFlow : public virtual ParameterizedFunction_TensorFlow<Dtype,
                                                                             stateDim + actionDim, 1>,
                             public virtual Qfunction<Dtype, stateDim, actionDim> {

 public:
  using QfunctionBase = Qfunction<Dtype, stateDim, actionDim>;
  using Pfunction_tensorflow = ParameterizedFunction_TensorFlow<Dtype, stateDim + actionDim, 1>;
  typedef typename QfunctionBase::State State;
  typedef typename QfunctionBase::StateBatch StateBatch;
  typedef typename QfunctionBase::Action Action;
  typedef typename QfunctionBase::ActionBatch ActionBatch;
  typedef typename QfunctionBase::Jacobian Jacobian;
  typedef typename QfunctionBase::Value Value;
  typedef typename QfunctionBase::ValueBatch ValueBatch;
  typedef Eigen::Matrix<Dtype, actionDim, Eigen::Dynamic> JacobianQwrtActionBatch;

  Qfunction_TensorFlow(std::string pathToGraphDefProtobuf, Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(pathToGraphDefProtobuf, learningRate) {
  }

  Qfunction_TensorFlow(std::string computeMode,
                       std::string graphName,
                       std::string graphParam,
                       Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(
          "Qfunction", computeMode, graphName, graphParam, learningRate) {
  }

  virtual void forward(State &state, Action &action, Dtype &value) {
    std::vector<MatrixXD> vectorOfOutputs;
    this->tf_->run({{"state", state},
                    {"action", action},
                    {"updateBNparams", this->notUpdateBN}},
                   {"QValue"}, {}, vectorOfOutputs);
    value = vectorOfOutputs[0](0);
  }

  virtual void forward(StateBatch &states, ActionBatch &actions, ValueBatch &values) {
    std::vector<MatrixXD> vectorOfOutputs;
    this->tf_->run({{"state", states},
                    {"action", actions},
                    {"updateBNparams", this->notUpdateBN}},
                   {"QValue"}, {}, vectorOfOutputs);
    values = vectorOfOutputs[0];
  }

  Dtype performOneSolverIter(StateBatch &states, ActionBatch &actions, ValueBatch &values) {
    std::vector<MatrixXD> outputs, dummy;
    this->tf_->run({{"state", states},
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

  Dtype performOneSolverIter_infimum(StateBatch &states, ActionBatch &actions, ValueBatch &values, Dtype linSlope) {
    std::vector<MatrixXD> outputs, dummy;
    auto slope = Eigen::Matrix<Dtype, 1, 1>::Constant(linSlope);
    this->tf_->run({{"state", states},
                    {"action", actions},
                    {"targetQValue", values},
                    {"trainUsingTargetQValue_infimum/linSlope", slope},
                    {"trainUsingTargetQValue_infimum/learningRate", this->learningRate_},
                    {"updateBNparams",
                     this->notUpdateBN}}, {"trainUsingTargetQValue_infimum/loss"},
                   {"trainUsingTargetQValue_infimum/solver"}, outputs);
    this->tf_->run({{"state", states},
                    {"action", actions},
                    {"updateBNparams",
                     this->updateBN}}, {},
                   {"QValue"}, dummy);
    return outputs[0](0);
  }

  Dtype getGradient_AvgOf_Q_wrt_action(StateBatch &states, ActionBatch &actions,
                                       JacobianQwrtActionBatch &gradients) const {
    std::vector<MatrixXD> outputs;
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