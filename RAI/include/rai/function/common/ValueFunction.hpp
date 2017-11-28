//
// Created by jemin on 27.07.16.
//

#ifndef RAI_VALUE_HPP
#define RAI_VALUE_HPP

#include <Eigen/Dense>
#include <Eigen/Core>

namespace rai {
namespace FuncApprox {

template<typename Dtype, int stateDimension, int actionDimension>
class ValueFunction : public virtual ParameterizedFunction <Dtype, stateDimension, 1>  {

public:
  using FunctionBase = ParameterizedFunction <Dtype, stateDimension, 1>;
  using Dataset = rai::Algorithm::LearningData<Dtype, stateDimension, actionDimension>;

  typedef typename FunctionBase::Input State;
  typedef typename FunctionBase::InputBatch StateBatch;
  typedef typename FunctionBase::Output Value;
  typedef typename FunctionBase::OutputBatch ValueBatch;
  typedef typename FunctionBase::Gradient Gradient;
  typedef typename FunctionBase::Jacobian Jacobian;
  typedef typename FunctionBase::Tensor1D Tensor1D;
  typedef typename FunctionBase::Tensor2D Tensor2D;
  typedef typename FunctionBase::Tensor3D Tensor3D;

  ValueFunction(){};
  virtual ~ValueFunction(){};

  virtual Dtype performOneSolverIter_trustregion(StateBatch &states, ValueBatch &targetOutputs, ValueBatch &old_prediction) {
    LOG(FATAL) << "NOT IMPLEMENTED";
    return Dtype(0);
  };
  virtual Dtype performOneSolverIter_trustregion(Dataset *minibatch, Tensor2D &old_prediction) {
    LOG(FATAL) << "NOT IMPLEMENTED";
    return Dtype(0);
  };
};

}} // namespaces

#endif //RAI_VALUE_HPP
