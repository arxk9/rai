//
// Created by joonho on 13.07.17.
//

#ifndef RAI_RECURRENTSTOCHASTICPOLICY_TENSORFLOW_HPP
#define RAI_RECURRENTSTOCHASTICPOLICY_TENSORFLOW_HPP
//
// Created by joonho on 23.03.17.
//


#include <rai/noiseModel/NormalDistributionNoise.hpp>
#include <rai/function/common/StochasticPolicy.hpp>
#include "rai/function/common/Policy.hpp"
#include "rai/function/common/Qfunction.hpp"
#include "Qfunction_TensorFlow.hpp"
#include "common/ParameterizedFunction_TensorFlow.hpp"
#include "DeterministicPolicy_TensorFlow.hpp"
#include "VectorHelper.hpp"

namespace RAI {
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
  using PolicyBase = Policy<Dtype, stateDim, actionDim>;
  using Pfunction_tensorflow = ParameterizedFunction_TensorFlow<Dtype, stateDim, actionDim>;
  using Qfunction_tensorflow = Qfunction_TensorFlow<Dtype, stateDim, actionDim>;
  using Qfunction_ = Qfunction<Dtype, stateDim, actionDim>;
  using Noise_ = Noise::NormalDistributionNoise<Dtype, actionDim>;

  typedef typename PolicyBase::State State;
  typedef typename PolicyBase::StateBatch StateBatch;
  using RecurrentState = Eigen::VectorXd;
  using RecurrentStateBatch = Eigen::MatrixXd;
  typedef typename PolicyBase::Action Action;
  typedef typename PolicyBase::ActionBatch ActionBatch;
  typedef typename PolicyBase::Gradient Gradient;
  typedef typename PolicyBase::Jacobian Jacobian;
  typedef typename PolicyBase::Jacobian JacobianWRTstate;
  typedef typename PolicyBase::StateTensor StateTensor;
  typedef typename PolicyBase::ActionTensor ActionTensor;

  RecurrentStochasticPolicy_TensorFlow(std::string pathToGraphDefProtobuf, Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(pathToGraphDefProtobuf, learningRate) {
  }

  RecurrentStochasticPolicy_TensorFlow(std::string computeMode,
                                       std::string graphName,
                                       std::string graphParam,
                                       Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(
          "RecurrentStochasticPolicy", computeMode, graphName, graphParam, learningRate) {
    hdim = this->getInnerStatesize();
    h.resize(hdim, 0);
  }

  void getdistribution(StateBatch &states, ActionBatch &means, Action &stdev) {
    std::vector<MatrixXD> vectorOfOutputs;
    this->tf_->run({{"state", states}}, {"action", "stdev"}, {}, vectorOfOutputs);
    means = vectorOfOutputs[0];
    stdev = vectorOfOutputs[1].col(0);
  }

  ///PPO
  virtual void PPOpg(StateTensor &states,
                     ActionTensor &action,
                     ActionTensor &actionNoise,
                     Advantages &advs,
                     Action &Stdev,
                     VectorXD &len,
                     VectorXD &grad) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;

    Tensor<Dtype, 2> hiddenState({hiddenStateDim(), states.dim(2)}, "h_init");
    hiddenState.setZero();
    Tensor<Dtype, 3> advsT(advs, {advs.cols(),1,1}, "advantage");
    Tensor<Dtype, 1> StdevT(Stdev, {Stdev.rows()}, "stdv_o");
    Tensor<Dtype, 3> lenT(len, {1,1,len.size()}, "length");
    Tensor<Dtype, 1> gradT(grad, {grad.rows()}, "Algo/PPO/Pg2");

    this->tf_->run({states, action, actionNoise, hiddenState, advsT, StdevT, lenT},
                   {"Algo/PPO/Pg"},
                   {},
                   vectorOfOutputs);
    std::memcpy(grad.data(), vectorOfOutputs[0].flat<Dtype>().data(), sizeof(Dtype) * grad.size());
  }
  virtual void PPOpg_kladapt(StateTensor  &states,
                             ActionTensor &action,
                             ActionTensor &actionNoise,
                             Advantages &advs,
                             Action &Stdev,
                             VectorXD &len,
                             VectorXD &grad) {
    Tensor<Dtype, 2> hiddenState({hiddenStateDim(), states.dim(2)}, "h_init");
    hiddenState.setZero();
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    Tensor<Dtype, 3> advsT(advs, {advs.cols(),1,1}, "advantage");
    Tensor<Dtype, 1> StdevT(Stdev, {Stdev.rows()}, "stdv_o");
    Tensor<Dtype, 3> lenT(len, {1,1,len.size()}, "length");
    Tensor<Dtype, 1> gradT(grad, {grad.rows()}, "Algo/PPO/Pg2");

    this->tf_->run({states, action, actionNoise, hiddenState, advsT, StdevT, lenT},
                   {"Algo/PPO/Pg2"},
                   {},
                   vectorOfOutputs);
    std::memcpy(grad.data(), vectorOfOutputs[0].flat<Dtype>().data(), sizeof(Dtype) * grad.size());
  }
  virtual Dtype PPOgetkl(Tensor<Dtype, 3> &states,
                         Tensor<Dtype, 3> &action,
                         Tensor<Dtype, 3> &actionNoise,
                         Action &Stdev,
                         VectorXD &len) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    Tensor<Dtype, 2> hiddenState({hiddenStateDim(), states.dim(2)}, "h_init");
    hiddenState.setZero();
    Tensor<Dtype, 1> StdevT(Stdev, {Stdev.rows()}, "stdv_o");
    Tensor<Dtype, 3> lenT(len, {1,1,len.size()}, "length");

    this->tf_->run({states, action, actionNoise, hiddenState, StdevT, lenT},
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

  virtual void setPPOparams(const Dtype &kl_coeff, const Dtype &ent_coeff, const Dtype &clip_param) {
    std::vector<MatrixXD> dummy;
    VectorXD input;
    input.resize(3);
    input << kl_coeff, ent_coeff, clip_param;
    this->tf_->run({{"PPO_params_placeholder", input}}, {}, {"PPO_param_assign_ops"}, dummy);
  }

  virtual void forward(State &state, Action &action) {
    std::vector<MatrixXD> vectorOfOutputs;
    this->tf_->forward({{"state", state}},
                       {"action"}, vectorOfOutputs);
    action = vectorOfOutputs[0];
  }

  virtual void forward(StateBatch &states, ActionBatch &actions) {
    std::vector<Tensor3D> vectorOfOutputs;
    // Step
    // input shape = [dim, #batch]
    // state shape = [state_size, #batch]
    tensorflow::Tensor stateTensor(this->tf_->getTensorFlowDataType(), tensorflow::TensorShape({states.cols(), 1, stateDim}));
    tensorflow::Tensor hTensor(this->tf_->getTensorFlowDataType(), tensorflow::TensorShape({states.cols(), hdim}));
    tensorflow::Tensor len_tensor(this->tf_->getTensorFlowDataType(), tensorflow::TensorShape({states.cols(), 1, 1}));
    VectorXD len_vec = VectorXD::Constant(states.cols(), 1);
    if (h.cols() != states.cols()) {
      h.resize(hdim, states.cols());
      h.setZero();
    }

    std::memcpy(len_tensor.flat<Dtype>().data(), len_vec.data(), sizeof(Dtype) * len_vec.size());
    std::memcpy(stateTensor.flat<Dtype>().data(), states.data(), sizeof(Dtype) * states.size());
    std::memcpy(hTensor.flat<Dtype>().data(), h.data(), sizeof(Dtype) * h.size());
    this->tf_->run({{"state", stateTensor}, {"h_init", hTensor}, {"length", len_tensor}}, {"action", "h_state"}, {}, vectorOfOutputs);
    std::memcpy(actions.data(), vectorOfOutputs[0].data(), sizeof(Dtype) * actions.size());
    std::memcpy(h.data(), vectorOfOutputs[1].data(), sizeof(Dtype) * h.size());
  }

  virtual void forward(StateTensor &states, ActionTensor &actions) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    this->tf_->forward({states}, {"action"}, vectorOfOutputs);
    actions = vectorOfOutputs[0];
  }

  virtual void test(StateTensor &states, ActionTensor &actions) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    this->tf_->forward({states}, {"action"}, vectorOfOutputs);
    actions = vectorOfOutputs[0];
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

  virtual void test(StateBatch &states, ActionBatch &actions) {
    std::vector<MatrixXD> vectorOfOutputs;

    this->tf_->forward({{"state", states}},
                       {"testout"}, vectorOfOutputs);
    actions = vectorOfOutputs[0];
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

    if (n < coldim) {
      h.block(0, n, hdim, coldim - n) =
          h.block(0, n + 1, hdim, coldim - n);
    }
    h.conservativeResize(hdim, coldim);
  }

  virtual int getInnerStatesize() {
    std::vector<MatrixXD> vectorOfOutputs;
    this->tf_->run({}, {"h_dim"}, {}, vectorOfOutputs);
    return vectorOfOutputs[0](0);
  }

  int hiddenStateDim (){ return hdim; }

 protected:
  using MatrixXD = typename TensorFlowNeuralNetwork<Dtype>::MatrixXD;
  using Tensor3D = Eigen::Tensor<Dtype, 3>;
  int hdim = 0;
  MatrixXD h;
};
}//namespace FuncApprox
}//namespace RAI

#endif //RAI_RECURRENTSTOCHASTICPOLICY_TENSORFLOW_HPP
