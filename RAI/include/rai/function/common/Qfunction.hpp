//
// Created by jemin on 27.07.16.
//

#ifndef RAI_QFUNCTION_HPP
#define RAI_QFUNCTION_HPP

#include <Eigen/Dense>
#include <Eigen/Core>

#include "ParameterizedFunction.hpp"
#include <rai/algorithm/common/DataStruct.hpp>

namespace rai {
namespace FuncApprox {

template<typename Dtype, int stateDimension, int actionDimension>
class Qfunction : public virtual ParameterizedFunction <Dtype, stateDimension + actionDimension, 1>  {

public:

  using FunctionBase = ParameterizedFunction <Dtype, stateDimension, 1>;
  using history = rai::Algorithm::history<Dtype,stateDimension, actionDimension>;
  using historyWithA = rai::Algorithm::historyWithAdvantage<Dtype,stateDimension, actionDimension>;

  typedef typename FunctionBase::Input StateAction;
  typedef typename FunctionBase::InputBatch StateActionBatch;
  typedef typename FunctionBase::Output Value;
  typedef typename FunctionBase::OutputBatch ValueBatch;
  typedef typename FunctionBase::Gradient Gradient;
  typedef typename FunctionBase::Jacobian Jacobian;
  typedef typename FunctionBase::Tensor1D Tensor1D;
  typedef typename FunctionBase::Tensor2D Tensor2D;
  typedef typename FunctionBase::Tensor3D Tensor3D;

  typedef Eigen::Matrix<Dtype, stateDimension, 1> State;
  typedef Eigen::Matrix<Dtype, stateDimension, Eigen::Dynamic> StateBatch;
  typedef Eigen::Matrix<Dtype, actionDimension, 1> Action;
  typedef Eigen::Matrix<Dtype, actionDimension, Eigen::Dynamic> ActionBatch;

  Qfunction(){};
  virtual ~Qfunction(){};

  virtual void forward(State& state, Action& action, Dtype& value) = 0;
  virtual void forward(StateBatch& states, ActionBatch& actions, ValueBatch &values) = 0;
  virtual Dtype performOneSolverIter(StateBatch& states, ActionBatch& actions, ValueBatch &values) = 0;
  virtual Dtype performOneSolverIter_infimum(StateBatch &states, ActionBatch &actions, ValueBatch &values, Dtype linSlope) {};
};

}} // namespaces

#endif //RAI_QFUNCTION_HPP
