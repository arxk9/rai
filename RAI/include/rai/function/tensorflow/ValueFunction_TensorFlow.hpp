#include "rai/function/common/ValueFunction.hpp"
#include "common/ParameterizedFunction_TensorFlow.hpp"

namespace rai {
namespace FuncApprox {

template<typename Dtype, int stateDim>
class ValueFunction_TensorFlow : public virtual ParameterizedFunction_TensorFlow<Dtype, stateDim, 1>,
                                 public virtual ValueFunction<Dtype, stateDim> {

 public:
  using ValueFunctionBase = ValueFunction<Dtype, stateDim>;
  using Pfunction_tensorflow = ParameterizedFunction_TensorFlow<Dtype, stateDim, 1>;

  typedef typename ValueFunctionBase::State State;
  typedef typename ValueFunctionBase::StateBatch StateBatch;
  typedef typename ValueFunctionBase::Value Value;
  typedef typename ValueFunctionBase::ValueBatch ValueBatch;
  typedef typename ValueFunctionBase::Gradient Gradient;
  typedef typename ValueFunctionBase::Jacobian Jacobian;
  typedef typename ValueFunctionBase::StateTensor StateTensor;

  ValueFunction_TensorFlow(std::string pathToGraphDefProtobuf, Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(pathToGraphDefProtobuf, learningRate) {
  }

  ValueFunction_TensorFlow(std::string computeMode,
                           std::string graphName,
                           std::string graphParam,
                           Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(
          "Vfunction", computeMode, graphName, graphParam, learningRate) {
  }

  ~ValueFunction_TensorFlow() {};

  virtual void forward(State &state, Dtype &value) {
    rai::Vector<MatrixXD> vectorOfOutputs;
    this->tf_->run({{"state", state},
                    {"updateBNparams", this->notUpdateBN}},
                   {"value"}, {}, vectorOfOutputs);
    value = vectorOfOutputs[0](0);
  }

  virtual void forward(StateBatch &states, ValueBatch &values) {
    rai::Vector<MatrixXD> vectorOfOutputs;
    this->tf_->run({{"state", states},
                    {"updateBNparams", this->notUpdateBN}},
                   {"value"}, {}, vectorOfOutputs);
    values = vectorOfOutputs[0];
  }

  virtual Dtype performOneSolverIter(StateBatch &states, ValueBatch &values) {
    rai::Vector<MatrixXD> loss, dummy;
    this->tf_->run({{"state", states},
                    {"targetValue", values},
                    {"trainUsingTargetValue/learningRate", this->learningRate_},
                    {"updateBNparams", this->notUpdateBN}},
                   {"trainUsingTargetValue/loss"},
                   {"trainUsingTargetValue/solver"}, loss);
    return loss[0](0);
  }

  virtual Dtype performOneSolverIter_trustregion(StateBatch &states, ValueBatch &values, ValueBatch &old_values) {
    rai::Vector<MatrixXD> loss, dummy;
    this->tf_->run({{"state", states},
                    {"targetValue", values},
                    {"predictedValue", old_values},
                    {"trainUsingTRValue/learningRate", this->learningRate_},
                    {"updateBNparams", this->notUpdateBN}},
                   {"trainUsingTRValue/loss"},
                   {"trainUsingTRValue/solver"}, loss);
    return loss[0](0);
  }

  virtual Dtype performOneSolverIter_infimum(StateBatch &states, ValueBatch &values, Dtype linSlope) {
    rai::Vector<MatrixXD> loss, dummy;
    auto slope = Eigen::Matrix<Dtype, 1, 1>::Constant(linSlope);
    this->tf_->run({{"state", states},
                    {"targetValue", values},
                    {"trainUsingTargetValue_inifimum/linSlope", slope},
                    {"trainUsingTargetValue_inifimum/learningRate", this->learningRate_},
                    {"updateBNparams", this->notUpdateBN}},
                   {"trainUsingTargetValue_inifimum/loss"},
                   {"trainUsingTargetValue_inifimum/solver"}, loss);
    return loss[0](0);
  }

  void setClipRate(const Dtype param_in){
    rai::Vector<MatrixXD> dummy;
    Eigen::VectorXd input;
    input << param_in;
    this->tf_->run({{"param_assign_placeholder", input}}, {}, {"clip_param_assign"}, dummy);

  }

 protected:
  using MatrixXD = typename TensorFlowNeuralNetwork<Dtype>::MatrixXD;

};
} // namespace FuncApprox
} // namespace rai
