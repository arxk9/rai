//
// Created by jemin on 27.07.16.
//

#ifndef RAI_PARAMETERIZEDFUNCTION_HPP
#define RAI_PARAMETERIZEDFUNCTION_HPP

#include <Eigen/Dense>
#include <Eigen/Core>
#include <glog/logging.h>
#include <boost/type_traits/function_traits.hpp>
#include <third_party/eigen3/unsupported/Eigen/CXX11/Tensor>
#include "rai/RAI_Tensor.hpp"

namespace RAI {
namespace FuncApprox {

enum class LibraryID {
  notSpecified = 0,
  caffe,
  tensorFlow
};

template<typename Dtype, int inputDimension, int outputDimension>
class ParameterizedFunction {

 public:
  typedef Eigen::Matrix<Dtype, inputDimension, 1> Input;
  typedef Eigen::Matrix<Dtype, inputDimension, Eigen::Dynamic> InputBatch;
  typedef Eigen::Matrix<Dtype, outputDimension, 1> Output;
  typedef Eigen::Matrix<Dtype, outputDimension, Eigen::Dynamic> OutputBatch;
  typedef Eigen::Matrix<Dtype, outputDimension, inputDimension> Jacobian;
  typedef Eigen::Matrix<Dtype, inputDimension, 1> Gradient;
  typedef Eigen::Matrix<Dtype, outputDimension, Eigen::Dynamic> JacobianWRTparam;

  typedef Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic> InnerState;
  typedef RAI::Tensor<Dtype, 3> InputTensor;
  typedef RAI::Tensor<Dtype, 3> OutputTensor;
//  typedef std::vector<InputBatch> InputVector;
//  typedef std::vector<OutputBatch> OutputVector;

  typedef Eigen::Matrix<Dtype, Eigen::Dynamic, 1> Parameter;
  typedef Eigen::Matrix<Dtype, Eigen::Dynamic, 1> ParameterGradient;
  using Pfunction = ParameterizedFunction<Dtype, inputDimension, outputDimension>;

  ParameterizedFunction() {};
  virtual ~ParameterizedFunction() {};

  /// must be implemented all by function libraries

  virtual void forward(Input &input, Output &output) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void forward(InputBatch &intputs, OutputBatch &outputs) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void forward(InputTensor &intputs, OutputTensor &outputs) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual Dtype performOneSolverIter(InputBatch &states, OutputBatch &targetOutputs) {
    LOG(FATAL) << "NOT IMPLEMENTED";
    return Dtype(0);
  };

  virtual Dtype performOneSolverIter_infimum(InputBatch &states, OutputBatch &targetOutputs, Dtype linSlope) {
    LOG(FATAL) << "NOT IMPLEMENTED";
    return Dtype(0);
  };

  virtual Dtype performOneSolverIter_huber(InputBatch &states, OutputBatch &targetOutputs) {
    LOG(FATAL) << "NOT IMPLEMENTED";
    return Dtype(0);
  };
  virtual Dtype performOneSolverIter_trustregion(InputBatch &states, OutputBatch &targetOutputs, OutputBatch &old_prediction) {
    LOG(FATAL) << "NOT IMPLEMENTED";
    return Dtype(0);
  };
  virtual void backward(InputBatch &states, OutputBatch &targetOutputs, ParameterGradient &gradient) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void setLearningRate(Dtype LearningRate) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual Dtype getLearningRate() {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void copyStructureFrom(Pfunction const *referenceFunction) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void copyAPFrom(Pfunction const *referenceFunction) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void copyLPFrom(Pfunction const *referenceFunction) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void interpolateLPWith(Pfunction const *anotherFunction, Dtype ratio) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void interpolateAPWith(Pfunction const *anotherFunction, Dtype ratio) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual int getLPSize() { LOG(FATAL) << "NOT IMPLEMENTED"; };

  virtual int getAPSize() { LOG(FATAL) << "NOT IMPLEMENTED"; };

  virtual void getJacobian(Input &input, Jacobian &jacobian) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void getGradient(Input &input, Gradient &gradient) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void getJacobianOutputWRTparameter(Input &input, JacobianWRTparam &gradient) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void performOneStepTowardsGradient(OutputBatch &diff, InputBatch &input) {
    LOG(FATAL) << "NOT IMPLEMENTED";
  };

  virtual void getLP(Parameter &param) { LOG(FATAL) << "NOT IMPLEMENTED"; };

  virtual void getAP(Parameter &param) { LOG(FATAL) << "NOT IMPLEMENTED"; };

  virtual void setLP(Parameter &param) { LOG(FATAL) << "NOT IMPLEMENTED"; };

  virtual void setAP(Parameter &param) { LOG(FATAL) << "NOT IMPLEMENTED"; };

  /// get error from last batch
  virtual Dtype getLossFromLastBatch() { LOG(FATAL) << "NOT IMPLEMENTED"; };

  virtual void dumpParam(std::string fileName) { LOG(FATAL) << "NOT IMPLEMENTED"; };

  /// recurrent
  virtual bool isRecurrent() { LOG(FATAL) << "NOT IMPLEMENTED"; }
  virtual void reset(int n) { LOG(FATAL) << "NOT IMPLEMENTED"; }
  virtual void terminate(int n) { LOG(FATAL) << "NOT IMPLEMENTED"; }
  virtual int getInnerStatesize() { LOG(FATAL) << "NOT IMPLEMENTED"; }

  LibraryID libraryID_ = LibraryID::notSpecified;
  int parameterSize = 0;

};

}
} // namespaces

#endif //RAI_PARAMETERIZEDFUNCTION_HPP
