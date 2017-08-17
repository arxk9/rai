//
// Created by jhwangbo on 26/06/17.
//

#ifndef RAI_ADDITIVEPOLICY_HPP
#define RAI_ADDITIVEPOLICY_HPP

#include "StochasticPolicy.hpp"

namespace RAI {
namespace FuncApprox {

template<typename Dtype, int stateDim, int actionDim>
class AdditiveStochasticPolicy : public virtual StochasticPolicy<Dtype, stateDim, actionDim> {

 public:
  using Policy_ = StochasticPolicy<Dtype, stateDim, actionDim>;
  using Qfunction_ = Qfunction<Dtype, stateDim, actionDim>;
  typedef typename Policy_::State State;
  typedef typename Policy_::StateBatch StateBatch;
  typedef typename Policy_::Action Action;
  typedef typename Policy_::ActionBatch ActionBatch;
  typedef typename Policy_::Gradient Gradient;
  typedef typename Policy_::Jacobian Jacobian;
  typedef typename Policy_::JacobianWRTparam JacobianWRTparam;
  typedef typename Policy_::Jacobian JacobianWRTstate;
  typedef typename Policy_::JacoqWRTparam JacoqWRTparam;
  typedef Eigen::Matrix<Dtype, -1, 1> VectorXD;
  using Advantages = Eigen::Matrix<Dtype, 1, Eigen::Dynamic>;

  using PfunctionBase = ParameterizedFunction<Dtype, stateDim, actionDim>;
  typedef typename PfunctionBase::Parameter Parameter;

  AdditiveStochasticPolicy(Policy_ *staticPolicy, Policy_ *learnablePolicy) :
      staticPolicy_(staticPolicy), learnablePolicy_(learnablePolicy) {
  }

  virtual void getdistribution(StateBatch &states, ActionBatch &means, Action &stdev) {
    ActionBatch firstMean(means.rows(), means.cols());
    staticPolicy_->getdistribution(states, firstMean, stdev);
    learnablePolicy_->getdistribution(states, means, stdev);
    means = firstMean + means;
  }

  ///TRPO
  // singlesample
  virtual void TRPOpg(State &state,
                      Action &action,
                      Action &actionNoise,
                      Advantages &adv,
                      Action &Stdev,
                      VectorXD &grad) {
    Action firstAction, secondAction;
    staticPolicy_->forward(state, firstAction);
    secondAction = action - firstAction;
    learnablePolicy_->TRPOpg(state, secondAction, actionNoise, adv, Stdev, grad);
  }

  //batch
  virtual void TRPOpg(StateBatch &states,
                      ActionBatch &actionBat,
                      ActionBatch &actionNoise,
                      Advantages &advs,
                      Action &Stdev,
                      VectorXD &grad) {
    ActionBatch firstAction(actionBat.rows(), actionBat.cols()), secondAction(actionBat.rows(), actionBat.cols());
    staticPolicy_->forward(states, firstAction);
    secondAction = actionBat - firstAction;
    learnablePolicy_->TRPOpg(states, secondAction, actionNoise, advs, Stdev, grad);
  }

  virtual void TRPOfvp(StateBatch &states,
                       ActionBatch &actionBat,
                       ActionBatch &actionNoise,
                       Advantages &advs,
                       Action &Stdev,
                       VectorXD &grad, VectorXD &getfvp) {
    ActionBatch firstAction(actionBat.rows(), actionBat.cols()), secondAction(actionBat.rows(), actionBat.cols());
    staticPolicy_->forward(states, firstAction);
    secondAction = actionBat - firstAction;
    learnablePolicy_->TRPOfvp(states, secondAction, actionNoise, advs, Stdev, grad, getfvp);
  }

  virtual void TRPOcg(StateBatch &states,
                       ActionBatch &actionBat,
                       ActionBatch &actionNoise,
                       Advantages &advs,
                       Action &Stdev,
                       VectorXD &grad, VectorXD &getng) {
    ActionBatch firstAction(actionBat.rows(), actionBat.cols()), secondAction(actionBat.rows(), actionBat.cols());
    staticPolicy_->forward(states, firstAction);
    secondAction = actionBat - firstAction;
    learnablePolicy_->TRPOcg(states, secondAction, actionNoise, advs, Stdev, grad, getng);
  }


  virtual Dtype TRPOloss(StateBatch &states,
                         ActionBatch &actionBat,
                         ActionBatch &actionNoise,
                         Advantages &advs,
                         Action &Stdev) {
    ActionBatch firstAction(actionBat.rows(), actionBat.cols()), secondAction(actionBat.rows(), actionBat.cols());
    staticPolicy_->forward(states, firstAction);
    secondAction = actionBat - firstAction;
    learnablePolicy_->TRPOloss(states, secondAction, actionNoise, advs, Stdev);
  }

  ///PPO
  virtual void PPOpg(StateBatch &states,
                     ActionBatch &actionBat,
                     ActionBatch &actionNoise,
                     Advantages &advs,
                     Action &Stdev,
                     VectorXD &grad,
                     const Dtype kl_thres,
                     const Dtype penalty) {
    ActionBatch firstAction(actionBat.rows(), actionBat.cols()), secondAction(actionBat.rows(), actionBat.cols());
    staticPolicy_->forward(states, firstAction);
    secondAction = actionBat - firstAction;
    learnablePolicy_->PPOpg(states, secondAction, actionNoise, advs, Stdev, grad, kl_thres, penalty);
  }

  virtual Dtype PPOgetkl(StateBatch &states,
                         ActionBatch &actionBat,
                         ActionBatch &actionNoise,
                         Action &Stdev) {
    ActionBatch firstAction(actionBat.rows(), actionBat.cols()), secondAction(actionBat.rows(), actionBat.cols());
    staticPolicy_->forward(states, firstAction);
    secondAction = actionBat - firstAction;
    learnablePolicy_->PPOgetkl(states, secondAction, actionNoise, Stdev);
  }

  virtual Dtype PPOloss(StateBatch &states,
                        ActionBatch &actionBat,
                        ActionBatch &actionNoise,
                        Advantages &advs,
                        Action &Stdev, const Dtype kl_thres, const Dtype penalty) {
    ActionBatch firstAction(actionBat.rows(), actionBat.cols()), secondAction(actionBat.rows(), actionBat.cols());
    staticPolicy_->forward(states, firstAction);
    secondAction = actionBat - firstAction;
    learnablePolicy_->PPOloss(states, secondAction, actionNoise, advs, Stdev, kl_thres, penalty);
  }

  virtual void setStdev(const Action &Stdev) {
    learnablePolicy_->setStdev(Stdev);
  }

  virtual void getStdev(Action &Stdev) {
    learnablePolicy_->getStdev(Stdev);
  }

  virtual void forward(State &state, Action &action) {
    Action firstAction, secondAction;
    staticPolicy_->forward(state, firstAction);
    learnablePolicy_->forward(state, secondAction);
    action = firstAction + secondAction;
  }

  virtual void forward(StateBatch &states, ActionBatch &actions) {
    ActionBatch firstAction(actions.rows(), actions.cols()), secondAction(actions.rows(), actions.cols());
    staticPolicy_->forward(states, firstAction);
    learnablePolicy_->forward(states, secondAction);
    actions = firstAction + secondAction;
  }

  virtual Dtype performOneSolverIter(StateBatch &states, ActionBatch &actions) {
    ActionBatch firstAction(actions.rows(), actions.cols()), secondAction(actions.rows(), actions.cols());
    staticPolicy_->forward(states, firstAction);
    secondAction = actions - firstAction;
    learnablePolicy_->performOneSolverIter(states, secondAction);
  }

  virtual void trainUsingGrad(const VectorXD &grad) {
    learnablePolicy_->trainUsingGrad(grad);
  }

  virtual void trainUsingGrad(const VectorXD &grad, const Dtype learningrate) {
    learnablePolicy_->trainUsingGrad(grad, learningrate);
  }

  virtual void getJacobianAction_WRT_LP(State &state, JacobianWRTparam &jacobian) {
    learnablePolicy_->getJacobianAction_WRT_LP(state, jacobian);
  }

  virtual void getJacobianAction_WRT_State(State &state, JacobianWRTstate &jacobian) {
    learnablePolicy_->getJacobianAction_WRT_State(state, jacobian);
  }


  /// parameterized function methods

  virtual int getLPSize() {
    return learnablePolicy_->getLPSize();
  }

  virtual int getAPSize() {
    return learnablePolicy_->getAPSize();
  }

  virtual void getLP(Parameter &param) {
    learnablePolicy_->getLP(param);
  }

  virtual void setLP(Parameter &param) {
    learnablePolicy_->setLP(param);
  }

  virtual void getAP(Parameter &param) {
    learnablePolicy_->getAP(param);
  }

  virtual void setAP(Parameter &param) {
    learnablePolicy_->setAP(param);
  }

  virtual void copyLPFrom(PfunctionBase const *const refFcn) {
    learnablePolicy_->copyLPFrom(refFcn);
  }

  virtual void interpolateLPWith(PfunctionBase const *anotherFcn, Dtype ratio) {
    learnablePolicy_->interpolateLPWith(anotherFcn, ratio);
  }

  virtual void copyAPFrom(PfunctionBase const *const refFcn) {
    learnablePolicy_->copyAPFrom(refFcn);
  }

  virtual void interpolateAPWith(PfunctionBase const *anotherFcn, Dtype ratio) {
    learnablePolicy_->interpolateAPWith(anotherFcn, ratio);
  }

  virtual void setLearningRate(Dtype learningRate) {
    learnablePolicy_->setLearningRate(learningRate);
  }

  virtual Dtype getLearningRate() {
    return learnablePolicy_->getLearningRate();
  }

  virtual void dumpParam(std::string fileName) {
    learnablePolicy_->dumpParam(fileName);
  }


  Policy_ *staticPolicy_;
  Policy_ *learnablePolicy_;
};

}
}

#endif //RAI_ADDITIVEPOLICY_HPP
