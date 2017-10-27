//
// Created by jhwangbo on 26/06/17.
//

#ifndef RAI_DETERMINISTICPOLICY_HPP
#define RAI_DETERMINISTICPOLICY_HPP

#include <Eigen/Dense>
#include <Eigen/Core>

#include "Qfunction.hpp"
#include "ParameterizedFunction.hpp"
#include "Policy.hpp"

namespace rai {
namespace FuncApprox {

template<typename Dtype, int stateDim, int actionDim>
class DeterministicPolicy : public virtual Policy <Dtype, stateDim, actionDim> {

 public:

  using PolicyBase = Policy<Dtype, stateDim, actionDim>;
  using Qfunction_ = Qfunction<Dtype, stateDim, actionDim>;
  typedef typename PolicyBase::State State;
  typedef typename PolicyBase::StateBatch StateBatch;
  typedef typename PolicyBase::Action Action;
  typedef typename PolicyBase::ActionBatch ActionBatch;
  typedef typename PolicyBase::Gradient Gradient;
  typedef typename PolicyBase::Jacobian Jacobian;
  typedef typename PolicyBase::JacobianWRTparam JacobianWRTparam;
  typedef typename PolicyBase::Jacobian JacobianWRTstate;
  typedef typename PolicyBase::JacoqWRTparam JacoqWRTparam;

  typedef typename PolicyBase::Tensor1D Tensor1D;
  typedef typename PolicyBase::Tensor2D Tensor2D;
  typedef typename PolicyBase::Tensor3D Tensor3D;

  virtual void forward(State &state, Action &action) = 0;

  virtual void forward(StateBatch &states, ActionBatch &actions) = 0;

  virtual void forward(Tensor3D &states, Tensor3D &actions) =0;

  virtual Dtype performOneSolverIter(StateBatch &states, ActionBatch &actions) = 0;

  virtual Dtype backwardUsingCritic(Qfunction_ *qFunction, StateBatch &states) = 0;

  virtual Dtype getGradQwrtParam(Qfunction_ *qFunction, StateBatch &states, JacoqWRTparam &jaco) = 0;

  virtual void getJacobianAction_WRT_LP(State &state, JacobianWRTparam &jacobian) = 0;

  virtual void getJacobianAction_WRT_State(State &state, JacobianWRTstate &jacobian) = 0;

};

}} // namespaces

#endif //RAI_DETERMINISTICPOLICY_HPP
