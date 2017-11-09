//
// Created by jhwangbo on 26/06/17.
//

#ifndef RAI_STOCHASTICPOLICY_HPP
#define RAI_STOCHASTICPOLICY_HPP

#include <rai/noiseModel/NormalDistributionNoise.hpp>
#include "rai/function/common/Policy.hpp"
#include "rai/function/common/Qfunction.hpp"
#include <rai/algorithm/common/LearningData.hpp>

namespace rai {
namespace FuncApprox {

template<typename Dtype, int stateDim, int actionDim>
class StochasticPolicy : public virtual Policy<Dtype, stateDim, actionDim> {

 public:
  typedef Eigen::Matrix<Dtype, -1, 1> VectorXD;
  typedef Eigen::Matrix<Dtype, 2 * actionDim, -1> JacobianWRTparam;

  using PolicyBase = Policy<Dtype, stateDim, actionDim>;
  using Qfunction_ = Qfunction<Dtype, stateDim, actionDim>;
  using Noise_ = Noise::NormalDistributionNoise<Dtype, actionDim>;
  using LearningData_ = rai::Algorithm::LearningData<Dtype, stateDim, actionDim>;

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
  typedef typename LearningData_::tensorBatch TensorBatch_;

  virtual void getdistribution(StateBatch &states, ActionBatch &means, Action &stdev) = 0;

  ///TRPO
  virtual void TRPOpg(TensorBatch_ &minibatch,
                      Action &Stdev,
                      VectorXD &grad) { LOG(FATAL) << "Not implemented"; };

  virtual Dtype TRPOcg(TensorBatch_ &minibatch,
                       Action &Stdev,
                       VectorXD &grad, VectorXD &getng) {
    LOG(FATAL) << "Not implemented";
    return 0;
  }

  virtual Dtype TRPOloss(TensorBatch_ &minibatch,
                         Action &Stdev) {
    LOG(FATAL) << "Not implemented";
    return 0;
  }

  ///PPO
  virtual void PPOpg(TensorBatch_ &minibatch,
                     Action &Stdev,
                     VectorXD &grad) { LOG(FATAL) << "Not implemented"; }

  virtual void PPOpg_kladapt(TensorBatch_ &minibatch,
                     Action &Stdev,
                     VectorXD &grad) { LOG(FATAL) << "Not implemented"; }

  virtual Dtype PPOgetkl(TensorBatch_ &minibatch,
                         Action &Stdev) {
    LOG(FATAL) << "Not implemented";
    return 0;
  }


  virtual void test(TensorBatch_ &minibatch,
                    Action &Stdev) {

  }
  /// common
  virtual void setStdev(const Action &Stdev) = 0;

  virtual void getStdev(Action &Stdev) = 0;

  virtual void setPPOparams(const Dtype &kl_coeff, const Dtype &ent_coeff, const Dtype &clip_param) {
    LOG(FATAL) << "Not implemented";
  }

  virtual void forward(State &state, Action &action) =0;
  virtual void forward(Tensor3D &states, Tensor3D &actions) =0;

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
};
}//namespace FuncApprox
}//namespace rai


#endif //RAI_STOCHASTICPOLICY_HPP
