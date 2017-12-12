//
// Created by joonho on 13.07.17.
//

#ifndef RAI_RECURRENTSTOCHASTICPOLICY_TENSORFLOW_HPP
#define RAI_RECURRENTSTOCHASTICPOLICY_TENSORFLOW_HPP


#include <rai/function/common/StochasticPolicy.hpp>
#include "common/RecurrentParametrizedFunction_TensorFlow.hpp"
#include "rai/common/VectorHelper.hpp"

namespace rai {
namespace FuncApprox {

template<typename Dtype, int stateDim, int actionDim>
class RecurrentStochasticPolicy_TensorFlow : public virtual StochasticPolicy<Dtype, stateDim, actionDim>,
                                             public virtual RecurrentParameterizedFunction_TensorFlow<Dtype,
                                                                                             stateDim,
                                                                                             actionDim> {
 public:
  typedef Eigen::Matrix<Dtype, -1, 1> VectorXD;
  typedef Eigen::Matrix<Dtype, -1, -1> MatrixXD;
  typedef Eigen::Matrix<Dtype, 2 * actionDim, -1> JacobianWRTparam;

  using Advantages = Eigen::Matrix<Dtype, 1, Eigen::Dynamic>;
  using PolicyBase = StochasticPolicy<Dtype, stateDim, actionDim>;
  using Pfunction_tensorflow = RecurrentParameterizedFunction_TensorFlow<Dtype, stateDim, actionDim>;

  using Pfunction_tensorflow::h;
  using Pfunction_tensorflow::hdim;

  typedef Eigen::Map<Eigen::Matrix<Dtype, -1, -1>> EigenMat;
  typedef typename PolicyBase::State State;
  typedef typename PolicyBase::Action Action;
  typedef typename PolicyBase::Gradient Gradient;
  typedef typename PolicyBase::Jacobian Jacobian;

  typedef typename PolicyBase::Tensor1D Tensor1D;
  typedef typename PolicyBase::Tensor2D Tensor2D;
  typedef typename PolicyBase::Tensor3D Tensor3D;
  typedef typename PolicyBase::Dataset Dataset;

  RecurrentStochasticPolicy_TensorFlow(std::string pathToGraphDefProtobuf, Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::RecurrentParameterizedFunction_TensorFlow(pathToGraphDefProtobuf, learningRate) {
  }

  RecurrentStochasticPolicy_TensorFlow(std::string computeMode,
                                       std::string graphName,
                                       std::string graphParam,
                                       Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::RecurrentParameterizedFunction_TensorFlow("testfunction", computeMode, graphName, graphParam, learningRate) {

  }

  ///PPO
  virtual void PPOpg(Dataset *minibatch,
                     Action &Stdev,
                     VectorXD &grad) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    Tensor1D StdevT(Stdev, {Stdev.rows()}, "stdv_o");
    Tensor2D hiddenState({hiddenStateDim(), minibatch->states.batches()}, 0, "h_init");

    this->tf_->run({minibatch->states,
                    minibatch->actions,
                    minibatch->actionNoises,
                    minibatch->advantages,
                    minibatch->lengths,
                    hiddenState, StdevT},
                   {"Algo/PPO/Pg"},
                   {},
                   vectorOfOutputs);
    std::memcpy(grad.data(), vectorOfOutputs[0].flat<Dtype>().data(), sizeof(Dtype) * grad.size());
  }
  virtual void PPOpg_kladapt(Dataset *minibatch,
                             Action &Stdev,
                             VectorXD &grad) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    Tensor1D StdevT(Stdev, {Stdev.rows()}, "stdv_o");
    Tensor2D hiddenState({hiddenStateDim(), minibatch->states.batches()},0, "h_init");

    this->tf_->run({minibatch->states,
                    minibatch->actions,
                    minibatch->actionNoises,
                    minibatch->advantages,
                    minibatch->lengths,
                    hiddenState, StdevT},
                   {"Algo/PPO/Pg2"},
                   {},
                   vectorOfOutputs);
    std::memcpy(grad.data(), vectorOfOutputs[0].template flat<Dtype>().data(), sizeof(Dtype) * grad.size());
  }
  virtual Dtype PPOgetkl(Dataset *minibatch,
                         Action &Stdev) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    Tensor1D StdevT(Stdev, {Stdev.rows()}, "stdv_o");
    Tensor2D hiddenState({hiddenStateDim(),  minibatch->states.batches()},0, "h_init");

    this->tf_->run({minibatch->states,
                    minibatch->actions,
                    minibatch->actionNoises,
                    minibatch->lengths,
                    hiddenState, StdevT},
                   {"Algo/PPO/kl_mean"},
                   {},
                   vectorOfOutputs);
    return vectorOfOutputs[0].flat<Dtype>().data()[0];
  }

  virtual void setStdev(const Action &Stdev) {
    std::vector<MatrixXD> dummy;
    this->tf_->run({{"Stdev_placeholder", Stdev}}, {}, {"assignStdev"}, dummy);
  }

  virtual void getStdev(Action &Stdev) {
    std::vector<MatrixXD> vectorOfOutputs;
    this->tf_->run({{"Stdev_placeholder", Stdev}}, {"getStdev"}, {}, vectorOfOutputs);
    Stdev = vectorOfOutputs[0];
  }

  virtual void setParams(const VectorXD params) {
    std::vector<MatrixXD> dummy;
    this->tf_->run({{"PPO_params_placeholder", params}}, {}, {"PPO_param_assign_ops"}, dummy);
  }

  virtual void trainUsingGrad(const VectorXD &grad, const Dtype learningrate) {
    std::vector<MatrixXD> dummy;
    VectorXD inputrate(1);
    inputrate(0) = learningrate;
    this->tf_->run({{"trainUsingGrad/Inputgradient", grad},
                    {"trainUsingGrad/learningRate", inputrate}}, {},
                   {"trainUsingGrad/applyGradients"}, dummy);
  }
};
}//namespace FuncApprox
}//namespace rai

#endif //RAI_RECURRENTSTOCHASTICPOLICY_TENSORFLOW_HPP
