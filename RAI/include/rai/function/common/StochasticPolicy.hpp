//
// Created by jhwangbo on 26/06/17.
//

#ifndef RAI_STOCHASTICPOLICY_HPP
#define RAI_STOCHASTICPOLICY_HPP

#include <rai/noiseModel/NormalDistributionNoise.hpp>
#include "rai/function/common/Policy.hpp"
#include "rai/function/common/Qfunction.hpp"

namespace RAI {
namespace FuncApprox {

template<typename Dtype, int stateDim, int actionDim>
class StochasticPolicy : public virtual Policy<Dtype, stateDim, actionDim> {

 public:
  typedef Eigen::Matrix<Dtype, -1, 1> VectorXD;
  typedef Eigen::Matrix<Dtype, 2 * actionDim, -1> JacobianWRTparam;

  using Advantages = Eigen::Matrix<Dtype, 1, Eigen::Dynamic>;
  using PolicyBase = Policy<Dtype, stateDim, actionDim>;
  using Qfunction_ = Qfunction<Dtype, stateDim, actionDim>;
  using Noise_ = Noise::NormalDistributionNoise<Dtype, actionDim>;

  typedef typename PolicyBase::State State;
  typedef typename PolicyBase::StateBatch StateBatch;
  typedef typename PolicyBase::Action Action;
  typedef typename PolicyBase::ActionBatch ActionBatch;
  typedef typename PolicyBase::Gradient Gradient;
  typedef typename PolicyBase::Jacobian Jacobian;
  typedef typename PolicyBase::Jacobian JacobianWRTstate;
  typedef typename PolicyBase::StateTensor StateTensor;
  typedef typename PolicyBase::ActionTensor ActionTensor;
//  typedef typename PolicyBase::InputVector StateVector;
//  typedef typename PolicyBase::OutputVector ActionVector;

  virtual void getdistribution(StateBatch &states, ActionBatch &means, Action &stdev) = 0;

  ///TRPO
  // singlesample
  virtual void TRPOpg(State &state,
                      Action &action,
                      Action &actionNoise,
                      Advantages &adv,
                      Action &Stdev,
                      VectorXD &grad) { LOG(FATAL) << "Not implemented"; };

  //batch
  virtual void TRPOpg(StateBatch &states,
                      ActionBatch &actionBat,
                      ActionBatch &actionNoise,
                      Advantages &advs,
                      Action &Stdev,
                      VectorXD &grad) { LOG(FATAL) << "Not implemented"; };

  virtual void TRPOfvp(StateBatch &states,
                       ActionBatch &actionBat,
                       ActionBatch &actionNoise,
                       Advantages &advs,
                       Action &Stdev,
                       VectorXD &grad, VectorXD &getfvp) { LOG(FATAL) << "Not implemented"; }

  virtual Dtype TRPOcg(StateBatch &states,
                       ActionBatch &actionBat,
                       ActionBatch &actionNoise,
                       Advantages &advs,
                       Action &Stdev,
                       VectorXD &grad, VectorXD &getng) { LOG(FATAL) << "Not implemented"; return 0;}

  virtual Dtype TRPOloss(StateBatch &states,
                       ActionBatch &actionBat,
                       ActionBatch &actionNoise,
                       Advantages &advs,
                       Action &Stdev) { LOG(FATAL) << "Not implemented"; return 0;}

  ///PPO
  virtual void PPOpg(StateBatch &states,
                     ActionBatch &actionBat,
                     ActionBatch &actionNoise,
                     Advantages &advs,
                     Action &Stdev,
                     VectorXD &grad) { LOG(FATAL) << "Not implemented"; }

  virtual void PPOpg_kladapt(StateBatch &states,
                             ActionBatch &actionBat,
                             ActionBatch &actionNoise,
                             Advantages &advs,
                             Action &Stdev,
                             VectorXD &grad) { LOG(FATAL) << "Not implemented"; }

  virtual Dtype PPOgetkl(StateBatch &states,
                         ActionBatch &actionBat,
                         ActionBatch &actionNoise,
                         Action &Stdev) { LOG(FATAL) << "Not implemented";  return 0; }

  ///recurrent
  virtual void PPOpg(RAI::Tensor<Dtype, 3> &states,
                     RAI::Tensor<Dtype, 3> &action,
                     RAI::Tensor<Dtype, 3> &actionNoise,
                     Advantages &advs,
                     Action &Stdev,
                     VectorXD &len,
                     VectorXD &grad) { LOG(FATAL) << "Not implemented"; }

  virtual void PPOpg_kladapt(RAI::Tensor<Dtype, 3> &states,
                             RAI::Tensor<Dtype, 3> &action,
                             RAI::Tensor<Dtype, 3> &actionNoise,
                             Advantages &advs,
                             Action &Stdev,
                             VectorXD &len,
                             VectorXD &grad) { LOG(FATAL) << "Not implemented"; }

  virtual Dtype PPOgetkl(RAI::Tensor<Dtype, 3> &states,
                         RAI::Tensor<Dtype, 3> &actionBat,
                         RAI::Tensor<Dtype, 3> &actionNoise,
                         Action &Stdev,
                         VectorXD &len) {LOG(FATAL) << "Not implemented"; return 0; }


  /// common

  virtual void setStdev(const Action &Stdev) = 0;

  virtual void getStdev(Action &Stdev) = 0;

  virtual void setPPOparams(const Dtype &kl_coeff, const Dtype &ent_coeff, const Dtype &clip_param) {
    LOG(FATAL) << "Not implemented";
  }

  virtual void forward(State &state, Action &action) =0;
  virtual void forward(StateBatch &states, ActionBatch &actions) =0;

  virtual Dtype performOneSolverIter(StateBatch &states, ActionBatch &actions) {
    LOG(FATAL) << "Not implemented";
    return 0;
  }

  virtual void trainUsingGrad(const VectorXD &grad) { LOG(FATAL) << "Not implemented"; };

  virtual void trainUsingGrad(const VectorXD &grad, const Dtype learningrate) { LOG(FATAL) << "Not implemented"; }

  virtual void getJacobianAction_WRT_LP(State &state, JacobianWRTparam &jacobian) { LOG(FATAL) << "Not implemented"; }

  virtual void getJacobianAction_WRT_State(State &state, JacobianWRTstate &jacobian) {
    LOG(FATAL) << "Not implemented";
  }

  virtual bool isRecurrent() { return false; }
  virtual void reset(int n) {}
  virtual void terminate(int n) {}
  virtual int getInnerStatesize() { return 0; }
};
}//namespace FuncApprox
}//namespace RAI


#endif //RAI_STOCHASTICPOLICY_HPP
