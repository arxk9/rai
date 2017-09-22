//
// Created by jemin on 27.07.16.
//

#ifndef RAI_POLICY_HPP
#define RAI_POLICY_HPP

#include <Eigen/Dense>
#include <Eigen/Core>

#include "Qfunction.hpp"
#include "ParameterizedFunction.hpp"

namespace rai {
namespace FuncApprox {

template<typename Dtype, int stateDimension, int actionDimension>
class Policy : public virtual ParameterizedFunction <Dtype, stateDimension, actionDimension> {

public:

  using FunctionBase = ParameterizedFunction <Dtype, stateDimension, actionDimension>;

  typedef typename FunctionBase::Input State;
  typedef typename FunctionBase::InputBatch StateBatch;
  typedef typename FunctionBase::Output Action;
  typedef typename FunctionBase::OutputBatch ActionBatch;
  typedef typename FunctionBase::InputTensor StateTensor;
  typedef typename FunctionBase::OutputTensor ActionTensor;

  typedef typename FunctionBase::Gradient Gradient;
  typedef typename FunctionBase::Jacobian Jacobian;
  typedef typename FunctionBase::JacobianWRTparam JacobianWRTparam;
  typedef Eigen::Matrix<Dtype, 1, Eigen::Dynamic> JacoqWRTparam;
  using Qfunction_ =  Qfunction<Dtype, stateDimension, actionDimension>;

  Policy(){};
  virtual ~Policy(){};

  virtual int hiddenStateDim() {return 0;};

};

}} // namespaces
#endif //RAI_POLICY_HPP
