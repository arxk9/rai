//
// Created by joonho on 13.07.17.
//

#ifndef RAI_RECURRENTSTOCHASTICPOLICY_TENSORFLOW_HPP
#define RAI_RECURRENTSTOCHASTICPOLICY_TENSORFLOW_HPP


#include <rai/function/common/StochasticPolicy.hpp>
#include "common/ParameterizedFunction_TensorFlow.hpp"
#include "rai/common/VectorHelper.hpp"

namespace rai {
namespace FuncApprox {

template<typename Dtype, int stateDim, int actionDim>
class RecurrentStochasticPolicy_TensorFlow : public virtual StochasticPolicy<Dtype, stateDim, actionDim>,
                                             public virtual ParameterizedFunction_TensorFlow<Dtype,
                                                                                             stateDim,
                                                                                             actionDim> {
 public:
  typedef Eigen::Matrix<Dtype, -1, 1> VectorXD;
  typedef Eigen::Matrix<Dtype, -1, -1> testMatrix;
  typedef Eigen::Matrix<Dtype, 2 * actionDim, 2 * actionDim> FimInActionSpace;
  typedef Eigen::Matrix<Dtype, 2 * actionDim, -1> JacobianWRTparam;

  using Advantages = Eigen::Matrix<Dtype, 1, Eigen::Dynamic>;
  using PolicyBase = StochasticPolicy<Dtype, stateDim, actionDim>;
  using Pfunction_tensorflow = ParameterizedFunction_TensorFlow<Dtype, stateDim, actionDim>;
  using Noise_ = Noise::NormalDistributionNoise<Dtype, actionDim>;
  
  typedef Eigen::Map<Eigen::Matrix<Dtype, -1, -1>> EigenMat;
  typedef typename PolicyBase::State State;
  typedef typename PolicyBase::StateBatch StateBatch;
  using RecurrentState = Eigen::VectorXd;
  using RecurrentStateBatch = Eigen::MatrixXd;
  typedef typename PolicyBase::Action Action;
  typedef typename PolicyBase::ActionBatch ActionBatch;
  typedef typename PolicyBase::Gradient Gradient;
  typedef typename PolicyBase::Jacobian Jacobian;
  typedef typename PolicyBase::Jacobian JacobianWRTstate;

  typedef typename PolicyBase::Tensor1D Tensor1D;
  typedef typename PolicyBase::Tensor2D Tensor2D;
  typedef typename PolicyBase::Tensor3D Tensor3D;
  typedef typename PolicyBase::Dataset Dataset;

  RecurrentStochasticPolicy_TensorFlow(std::string pathToGraphDefProtobuf, Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(pathToGraphDefProtobuf, learningRate) {
  }

  RecurrentStochasticPolicy_TensorFlow(std::string computeMode,
                                       std::string graphName,
                                       std::string graphParam,
                                       Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(
          "RecurrentStochasticPolicy", computeMode, graphName, graphParam, learningRate), h("h_init") {
    hdim = this->getHiddenStatesize();
    h.resize(hdim, 0);
  }

  void getdistribution(StateBatch &states, ActionBatch &means, Action &stdev) {
    std::vector<MatrixXD> vectorOfOutputs;
    this->tf_->run({{"state", states}}, {"action", "stdev"}, {}, vectorOfOutputs);
    means = vectorOfOutputs[0];
    stdev = vectorOfOutputs[1].col(0);
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

  virtual void forward(State &state, Action &action) {
    std::vector<MatrixXD> vectorOfOutputs;
    MatrixXD h_, length;
    length.resize(1,1);
    if (h.cols() != 1) {
      h_.resize(hdim, 1);
      h.setZero();
    }
    h_ = h.eMat();

    this->tf_->forward({{"state", state},
                        {"length", length},
                        {"h_init", h_}},
                       {"action",  "h_state"}, vectorOfOutputs);
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

  ///
  virtual void forward(Tensor3D &states, Tensor3D &actions) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    Tensor1D len({states.batches()},  states.dim(1), "length");

    if (h.cols() != states.batches()) {
      h.resize(hdim, states.batches());
      h.setZero();
    }
    this->tf_->run({states, h, len}, {"action", "h_state"}, {}, vectorOfOutputs);
    h.copyDataFrom(vectorOfOutputs[1]);
    actions.copyDataFrom(vectorOfOutputs[0]);
  }

  virtual void trainUsingGrad(const VectorXD &grad) {
    std::vector<MatrixXD> dummy;
    this->tf_->run({{"trainUsingGrad/Inputgradient", grad},
                    {"trainUsingGrad/learningRate", this->learningRate_}}, {},
                   {"trainUsingGrad/applyGradients"}, dummy);
  }

  virtual Dtype performOneSolverIter(StateBatch &states, ActionBatch &actions) { return 0; }
  virtual void trainUsingGrad(const VectorXD &grad, const Dtype learningrate) {
    std::vector<MatrixXD> dummy;
    VectorXD inputrate(1);
    inputrate(0) = learningrate;
    this->tf_->run({{"trainUsingGrad/Inputgradient", grad},
                    {"trainUsingGrad/learningRate", inputrate}}, {},
                   {"trainUsingGrad/applyGradients"}, dummy);
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
    LOG_IF(FATAL, coldim < 0) << "Initialize Hiddenstates first (Call reset)";
    LOG_IF(FATAL, n > coldim) << "n exceeds batchsize" << n << "vs." << coldim;
    h.removeCol(n);
  }

  virtual int getHiddenStatesize() {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    this->tf_->run({}, {"h_dim"}, {}, vectorOfOutputs);
    return vectorOfOutputs[0].scalar<int>()();
  }

  virtual void getHiddenStates(Tensor2D &h_out){
    h_out = h;
  }
  virtual typename EigenMat::ColXpr getHiddenState(int Id){
    return h.col(Id);
  }

  int hiddenStateDim() { return hdim; }

 protected:
  using MatrixXD = typename TensorFlowNeuralNetwork<Dtype>::MatrixXD;
  int hdim = 0;
  Tensor2D h;
};
}//namespace FuncApprox
}//namespace rai

#endif //RAI_RECURRENTSTOCHASTICPOLICY_TENSORFLOW_HPP
