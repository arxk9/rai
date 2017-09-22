//
// Created by jhwangbo on 01/09/17.
//

#ifndef RAI_DETERMINISTICMODEL_TENSORFLOW_HPP
#define RAI_DETERMINISTICMODEL_TENSORFLOW_HPP

#include "rai/function/common/DeterministicModel.hpp"

#pragma once

namespace rai {
namespace FuncApprox {

template<typename Dtype, int inputDim, int outputDim>
class DeterministicModel_TensorFlow : public virtual DeterministicModel<Dtype, inputDim, outputDim>,
                                      public virtual ParameterizedFunction_TensorFlow<Dtype, inputDim, outputDim> {

  using ModelBase = DeterministicModel<Dtype, inputDim, outputDim>;
  using Pfunction_tensorflow = ParameterizedFunction_TensorFlow<Dtype, inputDim, outputDim>;

  typedef typename ModelBase::Input Input;
  typedef typename ModelBase::InputBatch InputBatch;
  typedef typename ModelBase::Output Output;
  typedef typename ModelBase::OutputBatch OutputBatch;

 public:

  DeterministicModel_TensorFlow(std::string pathToGraphDefProtobuf, Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(pathToGraphDefProtobuf, learningRate) {
  }

  DeterministicModel_TensorFlow(std::string computeMode,
                                std::string graphName,
                                std::string graphParam,
                                Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(
          "DeterministicModel", computeMode, graphName, graphParam, learningRate) {
  }

  virtual void forward(Input &input, Output &output) {
    rai::Vector<MatrixXD> vectorOfOutputs;
    this->tf_->forward({{"input", input}},
                       {"output"}, vectorOfOutputs);
    output = vectorOfOutputs[0];
  }

  virtual void forward(InputBatch &inputs, OutputBatch &outputs) {
    rai::Vector<MatrixXD> vectorOfOutputs;
    this->tf_->forward({{"input", inputs}},
                       {"output"}, vectorOfOutputs);
    outputs = vectorOfOutputs[0];
  }

  virtual Dtype performOneSolverIter(InputBatch &inputs, OutputBatch &outputs) {
    rai::Vector<MatrixXD> loss, dummy;
    this->tf_->run({{"input", inputs},
                    {"targetOutput", outputs},
                    {"squareLoss/learningRate", this->learningRate_}}, {"squareLoss/loss"},
                   {"squareLoss/solver"}, loss);
    return loss[0](0);
  }
 protected:
  using MatrixXD = typename TensorFlowNeuralNetwork<Dtype>::MatrixXD;

};

}
}

#endif //RAI_DETERMINISTICMODEL_TENSORFLOW_HPP
