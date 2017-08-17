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

namespace RAI {
namespace FuncApprox {

template<typename Dtype, int stateDim, int actionDim>
class StochasticPolicy_TensorFlow : public virtual StochasticPolicy<Dtype, stateDim, actionDim>,
                                    public virtual ParameterizedFunction_TensorFlow<Dtype, stateDim, actionDim> {
 public:
  typedef Eigen::Matrix<Dtype, -1, 1> VectorXD;
  typedef Eigen::Matrix<Dtype, -1, -1> testMatrix;
  typedef Eigen::Matrix<Dtype, 2 * actionDim, 2 * actionDim> FimInActionSpace;
  typedef Eigen::Matrix<Dtype, 2 * actionDim, -1> JacobianWRTparam;

  using Advantages = Eigen::Matrix<Dtype, 1, Eigen::Dynamic>;
  using PolicyBase = StochasticPolicy<Dtype, stateDim, actionDim>;
  using Pfunction_tensorflow = ParameterizedFunction_TensorFlow<Dtype, stateDim, actionDim>;
  using Qfunction_tensorflow = Qfunction_TensorFlow<Dtype, stateDim, actionDim>;
  using Qfunction_ = Qfunction<Dtype, stateDim, actionDim>;
  using Noise_ = Noise::NormalDistributionNoise<Dtype, actionDim>;

  typedef typename PolicyBase::State State;
  typedef typename PolicyBase::StateBatch StateBatch;
  typedef typename PolicyBase::Action Action;
  typedef typename PolicyBase::ActionBatch ActionBatch;
  typedef typename PolicyBase::StateTensor StateTensor;
  typedef typename PolicyBase::ActionTensor ActionTensor;

  typedef typename PolicyBase::Gradient Gradient;
  typedef typename PolicyBase::Jacobian Jacobian;
  typedef typename PolicyBase::Jacobian JacobianWRTstate;

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

  // singlesample
  virtual void TRPOpg(State &state,
                      Action &action,
                      Action &actionNoise,
                      Advantages &adv,
                      Action &Stdev,
                      VectorXD &grad) {
    std::vector<MatrixXD> vectorOfOutputs;
    this->tf_->run({{"state", state},
                    {"sampled_oa", action},
                    {"noise_oa", actionNoise},
                    {"advantage", adv},
                    {"stdv_o", Stdev}
                   },
                   {"Algo/TRPO/Pg"},
                   {},
                   vectorOfOutputs);
    grad = vectorOfOutputs[0];
  }

  //batch
  virtual void TRPOpg(StateBatch &states,
                      ActionBatch &actionBat,
                      ActionBatch &actionNoise,
                      Advantages &advs,
                      Action &Stdev,
                      VectorXD &grad) {
    std::vector<MatrixXD> vectorOfOutputs;
    this->tf_->run({{"state", states},
                    {"sampled_oa", actionBat},
                    {"noise_oa", actionNoise},
                    {"advantage", advs},
                    {"stdv_o", Stdev}
                   },
                   {"Algo/TRPO/Pg"},
                   {},
                   vectorOfOutputs);
    grad = vectorOfOutputs[0];
  }

  virtual void TRPOfvp(StateBatch &states,
                       ActionBatch &actionBat,
                       ActionBatch &actionNoise,
                       Advantages &advs,
                       Action &Stdev,
                       VectorXD &grad, VectorXD &getfvp) {
    std::vector<MatrixXD> vectorOfOutputs;
    this->tf_->run({{"state", states},
                    {"sampled_oa", actionBat},
                    {"noise_oa", actionNoise},
                    {"advantage", advs},
                    {"stdv_o", Stdev},
                    {"tangent", grad}
                   },
                   {"Algo/TRPO/fvp"}, {}, vectorOfOutputs);
    getfvp = vectorOfOutputs[0];
  }

  virtual Dtype TRPOcg(StateBatch &states,
                       ActionBatch &actionBat,
                       ActionBatch &actionNoise,
                       Advantages &advs,
                       Action &Stdev,
                       VectorXD &grad, VectorXD &getng) {
    std::vector<MatrixXD> vectorOfOutputs;
    this->tf_->run({{"state", states},
                    {"sampled_oa", actionBat},
                    {"noise_oa", actionNoise},
                    {"advantage", advs},
                    {"stdv_o", Stdev},
                    {"tangent", grad}
                   },
                   {"Algo/TRPO/Cg", "Algo/TRPO/Cgerror"}, {}, vectorOfOutputs);
    getng = vectorOfOutputs[0];
    return vectorOfOutputs[1](0);
  }
  virtual Dtype TRPOloss(StateBatch &states,
                         ActionBatch &actionBat,
                         ActionBatch &actionNoise,
                         Advantages &advs,
                         Action &Stdev) {
    std::vector<MatrixXD> loss;
    this->tf_->run({{"state", states},
                    {"sampled_oa", actionBat},
                    {"noise_oa", actionNoise},
                    {"advantage", advs},
                    {"stdv_o", Stdev}
                   },
                   {"Algo/TRPO/loss"},
                   {}, loss);
    return loss[0](0);
  }

  ///PPO
  virtual void PPOpg(StateBatch &states,
                     ActionBatch &actionBat,
                     ActionBatch &actionNoise,
                     Advantages &advs,
                     Action &Stdev,
                     VectorXD &grad) {
    std::vector<MatrixXD> vectorOfOutputs;

    this->tf_->run({{"state", states},
                    {"sampled_oa", actionBat},
                    {"noise_oa", actionNoise},
                    {"advantage", advs},
                    {"stdv_o", Stdev}
                   },
                   {"Algo/PPO/Pg"},
                   {},
                   vectorOfOutputs);
    grad = vectorOfOutputs[0];
  }
  virtual void PPOpg_kladapt(StateBatch &states,
                             ActionBatch &actionBat,
                             ActionBatch &actionNoise,
                             Advantages &advs,
                             Action &Stdev,
                             VectorXD &grad) {
    std::vector<MatrixXD> vectorOfOutputs;

    this->tf_->run({{"state", states},
                    {"sampled_oa", actionBat},
                    {"noise_oa", actionNoise},
                    {"advantage", advs},
                    {"stdv_o", Stdev}
                   },
                   {"Algo/PPO/Pg2"},
                   {},
                   vectorOfOutputs);
    grad = vectorOfOutputs[0];
  }

  virtual Dtype PPOgetkl(StateBatch &states,
                         ActionBatch &actionBat,
                         ActionBatch &actionNoise,
                         Action &Stdev) {
    std::vector<MatrixXD> vectorOfOutputs;

    this->tf_->run({{"state", states},
                    {"sampled_oa", actionBat},
                    {"noise_oa", actionNoise},
                    {"stdv_o", Stdev},
                   },
                   {"Algo/PPO/kl_mean"},
                   {},
                   vectorOfOutputs);

    return vectorOfOutputs[0](0);
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
    std::vector<MatrixXD> vectorOfOutputs;
    this->tf_->forward({{"state", states}},
                       {"action"}, vectorOfOutputs);
    actions = vectorOfOutputs[0];
  }

  virtual void forward(StateTensor &states, ActionTensor &actions) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    this->tf_->forward({states}, {"action"}, vectorOfOutputs);
    actions = vectorOfOutputs[0];
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
  virtual bool isRecurrent() {
    return false;
  }
 protected:
  using MatrixXD = typename TensorFlowNeuralNetwork<Dtype>::MatrixXD;
  using Tensor3D = Eigen::Tensor<Dtype, 3>;

};
}//namespace FuncApprox
}//namespace RAI

#endif //RAI_STOCHPOL_TENSORFLOW_HPP
