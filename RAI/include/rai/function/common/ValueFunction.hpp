//
// Created by jemin on 27.07.16.
//

#ifndef RAI_VALUE_HPP
#define RAI_VALUE_HPP

#include <Eigen/Dense>
#include <Eigen/Core>

#include "ParameterizedFunction.hpp"

namespace RAI {
namespace FuncApprox {

template<typename Dtype, int stateDimension>
class ValueFunction : public virtual ParameterizedFunction <Dtype, stateDimension, 1>  {

public:

  using FunctionBase = ParameterizedFunction <Dtype, stateDimension, 1>;

  typedef typename FunctionBase::Input State;
  typedef typename FunctionBase::InputBatch StateBatch;
  typedef typename FunctionBase::Output Value;
  typedef typename FunctionBase::OutputBatch ValueBatch;
  typedef typename FunctionBase::Gradient Gradient;
  typedef typename FunctionBase::Jacobian Jacobian;
  typedef typename FunctionBase::InputTensor StateTensor;
  ValueFunction(){};
  virtual ~ValueFunction(){};

  virtual Dtype performOneSolverIter_trustregion(StateBatch &states, ValueBatch &targetOutputs, ValueBatch &old_prediction) {
    LOG(FATAL) << "NOT IMPLEMENTED";
    return Dtype(0);
  };

};

}} // namespaces

#endif //RAI_VALUE_HPP
