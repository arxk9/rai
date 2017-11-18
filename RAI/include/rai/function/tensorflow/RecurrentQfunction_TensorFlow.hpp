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

  typedef typename QfunctionBase::Action Action;
  typedef typename QfunctionBase::ActionBatch ActionBatch;
  typedef typename QfunctionBase::Jacobian Jacobian;
  typedef typename QfunctionBase::Value Value;
  typedef typename QfunctionBase::ValueBatch ValueBatch;
  typedef typename QfunctionBase::Tensor1D Tensor1D;
  typedef typename QfunctionBase::Tensor2D Tensor2D;
  typedef typename QfunctionBase::Tensor3D Tensor3D;

  typedef typename Pfunction_tensorflow ::InnerState InnerState;
  typedef typename QfunctionBase::TensorBatch_ TensorBatch_;

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

  virtual void forward(StateBatch &states, ActionBatch &actions, ValueBatch &values) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    Tensor3D stateT({stateDim, 1, states.cols()}, "state");
    Tensor3D actionT({actionDim, 1, states.cols()}, "action");
    Tensor1D len({states.cols()}, 1, "length");
    stateT.copyDataFrom(states);

    if (h.cols() != states.cols()) {
      h.resize(hdim, states.cols());
      h.setZero();
    }

    this->tf_->run({stateT, actionT, h, len}, {"QValue", "h_state"}, {}, vectorOfOutputs);
    std::memcpy(values.data(), vectorOfOutputs[0].flat<Dtype>().data(), sizeof(Dtype) * values.size());
    h.copyDataFrom(vectorOfOutputs[1]);
  }

//  Dtype performOneSolverIter(std::vector<StateBatch> &states,
//                             std::vector<ActionBatch> &actions,
//                             std::vector<ValueBatch> &values) {
//    std::vector<MatrixXD> outputs, dummy;
//    this->tf_->run({{"state", states},
//                    {"new_rcrnt_state", rcrntStates},
//                    {"action", actions},
//                    {"targetQValue", values},
//                    {"trainUsingTargetQValue/learningRate", this->learningRate_},
//                    {"updateBNparams",
//                     this->notUpdateBN}}, {"trainUsingTargetQValue/loss"},
//                   {"trainUsingTargetQValue/solver"}, outputs);
//    this->tf_->run({{"state", states},
//                    {"action", actions},
//                    {"updateBNparams",
//                     this->updateBN}}, {},
//                   {"QValue"}, dummy);
//    return outputs[0](0);
//  }
//
//  Dtype getGradient_AvgOf_Q_wrt_action(StateBatch &states, RecurrentStateBatch &rcrntStates, ActionBatch &actions,
//                                       JacobianQwrtActionBatch &gradients) const {
//    std::vector<MatrixXD> outputs;
//    this->tf_->run({{"state", states},
//                    {"action", actions},
//                    {"updateBNparams", this->notUpdateBN}},
//                   {"gradient_AvgOf_Q_wrt_action", "average_Q_value"}, {}, outputs);
//    gradients = outputs[0];
//    return outputs[1](0);
//  }

  virtual int getInnerStatesize() {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    this->tf_->run({}, {"h_dim"}, {}, vectorOfOutputs);
    return vectorOfOutputs[0].scalar<int>()();
  }

  virtual void getInnerStates(InnerState &h_out){
    h_out = h;
  }

  int hiddenStateDim() { return hdim; }

 protected:
  using MatrixXD = typename TensorFlowNeuralNetwork<Dtype>::MatrixXD;
  int hdim = 0;
  Tensor2D h;
};
}
}