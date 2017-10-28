//
// Created by joonho on 23.03.17.
//

#ifndef RAI_STOCHPOL_TENSORFLOW_HPP
#define RAI_STOCHPOL_TENSORFLOW_HPP

#include <rai/noiseModel/NormalDistributionNoise.hpp>
#include "rai/function/common/StochasticPolicy.hpp"
#include "rai/function/common/Qfunction.hpp"
#include "Qfunction_TensorFlow.hpp"
#include "common/ParameterizedFunction_TensorFlow.hpp"
namespace rai {
namespace FuncApprox {

template<typename Dtype, int stateDim, int actionDim>
class StochasticPolicy_TensorFlow : public virtual StochasticPolicy<Dtype, stateDim, actionDim>,
                                    public virtual ParameterizedFunction_TensorFlow<Dtype, stateDim, actionDim> {
 public:
  typedef Eigen::Matrix<Dtype, -1, 1> VectorXD;
  typedef Eigen::Matrix<Dtype, 2 * actionDim, -1> JacobianWRTparam;

  using Advantages = Eigen::Matrix<Dtype, 1, Eigen::Dynamic>;
  using PolicyBase = StochasticPolicy<Dtype, stateDim, actionDim>;
  using Pfunction_tensorflow = ParameterizedFunction_TensorFlow<Dtype, stateDim, actionDim>;
  using Noise_ = Noise::NormalDistributionNoise<Dtype, actionDim>;

  typedef typename PolicyBase::State State;
  typedef typename PolicyBase::StateBatch StateBatch;
  typedef typename PolicyBase::Action Action;
  typedef typename PolicyBase::ActionBatch ActionBatch;

  typedef typename PolicyBase::Gradient Gradient;
  typedef typename PolicyBase::Jacobian Jacobian;
  typedef typename PolicyBase::Jacobian JacobianWRTstate;

  typedef typename PolicyBase::Tensor1D Tensor1D;
  typedef typename PolicyBase::Tensor2D Tensor2D;
  typedef typename PolicyBase::Tensor3D Tensor3D;

  StochasticPolicy_TensorFlow(std::string pathToGraphDefProtobuf, Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(pathToGraphDefProtobuf, learningRate) {
  }

  StochasticPolicy_TensorFlow(std::string computeMode,
                              std::string graphName,
                              std::string graphParam,
                              Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(
          "StochasticPolicy", computeMode, graphName, graphParam, learningRate) {
  }

  virtual void getdistribution(StateBatch &states, ActionBatch &means, Action &stdev) {
    std::vector<MatrixXD> vectorOfOutputs;
    this->tf_->run({{"state", states}}, {"action", "stdev"}, {}, vectorOfOutputs);
    means = vectorOfOutputs[0];
    stdev = vectorOfOutputs[1].col(0);
  }
  ///TRPO
  //batch
  virtual void TRPOpg(Tensor3D &states,
                      Tensor3D &actions,
                      Tensor3D &actionNoise,
                      Advantages &advs,
                      Action &Stdev,
                      VectorXD &grad) {
    Tensor1D StdevT(Stdev, {Stdev.rows()}, "stdv_o");
    Tensor1D advsT(advs, {advs.cols()}, "advantage");
    std::vector<tensorflow::Tensor> vectorOfOutputs;

    this->tf_->run({states,
                    actions,
                    actionNoise,
                    advsT,
                    StdevT},
                   {"Algo/TRPO/Pg"},
                   {},
                   vectorOfOutputs);
    std::memcpy(grad.data(), vectorOfOutputs[0].template flat<Dtype>().data(), sizeof(Dtype) * grad.size());
  }

  virtual Dtype TRPOcg(Tensor3D &states,
                       Tensor3D &actions,
                       Tensor3D &actionNoise,
                       Advantages &advs,
                       Action &Stdev,
                       VectorXD &grad, VectorXD &getng) {
    Tensor1D StdevT(Stdev, {Stdev.rows()}, "stdv_o");
    Tensor1D advsT(advs, {advs.cols()}, "advantage");
    Tensor1D gradT(grad, {grad.rows()}, "tangent");
    std::vector<tensorflow::Tensor> vectorOfOutputs;

    this->tf_->run({states,
                    actions,
                    actionNoise,
                    advsT,
                    StdevT,
                    gradT},
                   {"Algo/TRPO/Cg", "Algo/TRPO/Cgerror"}, {}, vectorOfOutputs);
    std::memcpy(getng.data(), vectorOfOutputs[0].template flat<Dtype>().data(), sizeof(Dtype) * getng.size());
//    Dtype loss = *(vectorOfOutputs[1].flat<Dtype>().data());
    Dtype loss = 0;
    return loss;
  }

  virtual Dtype TRPOloss(Tensor3D &states,
                         Tensor3D &actions,
                         Tensor3D &actionNoise,
                         Advantages &advs,
                         Action &Stdev) {
    Tensor1D StdevT(Stdev, {Stdev.rows()}, "stdv_o");
    Tensor1D advsT(advs, {advs.cols()}, "advantage");
    std::vector<tensorflow::Tensor> vectorOfOutputs;

    this->tf_->run({states,
                    actions,
                    actionNoise,
                    advsT,
                    StdevT},
                   {"Algo/TRPO/loss"},
                   {}, vectorOfOutputs);
    Dtype loss = *(vectorOfOutputs[0].flat<Dtype>().data());
    return loss;
  }

  ///PPO
  virtual void PPOpg(Tensor3D &states,
                     Tensor3D &actions,
                     Tensor3D &actionNoise,
                     Advantages &advs,
                     Action &Stdev,
                     Tensor1D &len,
                     VectorXD &grad) {
    Tensor1D StdevT(Stdev, {Stdev.rows()}, "stdv_o");
    Tensor1D advsT(advs, {advs.cols()}, "advantage");
    std::vector<tensorflow::Tensor> vectorOfOutputs;

    this->tf_->run({states,
                    actions,
                    actionNoise,
                    advsT,
                    StdevT},
                   {"Algo/PPO/Pg"},
                   {},
                   vectorOfOutputs);
    std::memcpy(grad.data(), vectorOfOutputs[0].template flat<Dtype>().data(), sizeof(Dtype) * grad.size());
  }
  virtual void PPOpg_kladapt(Tensor3D &states,
                             Tensor3D &action,
                             Tensor3D &actionNoise,
                             Advantages &advs,
                             Action &Stdev,
                             Tensor1D &len,
                             VectorXD &grad) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    Tensor1D StdevT(Stdev, {Stdev.rows()}, "stdv_o");
    Tensor1D advsT(advs, {advs.cols()}, "advantage");

    this->tf_->run({states, action, actionNoise, advsT, StdevT},
                   {"Algo/PPO/Pg2"},
                   {},
                   vectorOfOutputs);

    std::memcpy(grad.data(), vectorOfOutputs[0].template flat<Dtype>().data(), sizeof(Dtype) * grad.size());
  }

  virtual Dtype PPOgetkl(Tensor3D &states,
                         Tensor3D &actions,
                         Tensor3D &actionNoise,
                         Action &Stdev,
                         Tensor1D &len) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    Tensor1D StdevT(Stdev, {Stdev.rows()}, "stdv_o");

    this->tf_->run({states, actions, actionNoise, StdevT},
                   {"Algo/PPO/kl_mean"},
                   {},
                   vectorOfOutputs);
    return vectorOfOutputs[0].template flat<Dtype>().data()[0];
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

  virtual void forward(StateBatch &state, ActionBatch &action) {
    std::vector<MatrixXD> vectorOfOutputs;
    this->tf_->forward({{"state", state}},
                       {"action"}, vectorOfOutputs);
    action = vectorOfOutputs[0];
  }

  virtual void forward(Tensor3D &states, Tensor3D &actions) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    this->tf_->forward({states}, {"action"}, vectorOfOutputs);
    actions.copyDataFrom(vectorOfOutputs[0]);
  }

  virtual Dtype performOneSolverIter(StateBatch &states, ActionBatch &actions) {
    std::vector<MatrixXD> loss, dummy;
    this->tf_->run({{"state", states},
                    {"targetAction", actions},
                    {"trainUsingTargetAction/learningRate", this->learningRate_}}, {"trainUsingTargetAction/loss"},
                   {"trainUsingTargetAction/solver"}, loss);
    this->tf_->run({{"state", states}}, {},
                   {"action"}, dummy);
    return loss[0](0);
  }
  void trainUsingGrad(const VectorXD &grad) {
    std::vector<MatrixXD> dummy;
    this->tf_->run({{"trainUsingGrad/Inputgradient", grad},
                    {"trainUsingGrad/learningRate", this->learningRate_}}, {},
                   {"trainUsingGrad/applyGradients"}, dummy);
  }
  virtual void trainUsingGrad(const VectorXD &grad, const Dtype learningrate) {
    std::vector<MatrixXD> dummy;
    VectorXD inputrate(1);
    inputrate(0) = learningrate;
    this->tf_->run({{"trainUsingGrad/Inputgradient", grad},
                    {"trainUsingGrad/learningRate", inputrate}}, {},
                   {"trainUsingGrad/applyGradients"}, dummy);
  }
  virtual void getJacobianAction_WRT_LP(State &state, JacobianWRTparam &jacobian) {
    std::vector<MatrixXD> temp;
    this->tf_->run({{"state", state}}, {"jac_Action_wrt_Param"}, {}, temp);
    jacobian = temp[0];
  }

  virtual void getJacobianAction_WRT_State(State &state, JacobianWRTstate &jacobian) {
    std::vector<MatrixXD> temp;
    this->tf_->run({{"state", state}}, {"jac_Action_wrt_State"}, {}, temp);
    jacobian = temp[0];
  }

 protected:
  using MatrixXD = typename TensorFlowNeuralNetwork<Dtype>::MatrixXD;

};
}//namespace FuncApprox
}//namespace rai

#endif //RAI_STOCHPOL_TENSORFLOW_HPP
