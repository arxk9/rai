#include "rai/function/common/ValueFunction.hpp"
#include "common/ParameterizedFunction_TensorFlow.hpp"

namespace RAI {
namespace FuncApprox {

template<typename Dtype, int stateDim>
class RecurrentValueFunction_TensorFlow : public virtual ParameterizedFunction_TensorFlow<Dtype, stateDim, 1>,
                                          public virtual ValueFunction<Dtype, stateDim> {

 public:
  using ValueFunctionBase = ValueFunction<Dtype, stateDim>;
  using Pfunction_tensorflow = ParameterizedFunction_TensorFlow<Dtype, stateDim, 1>;

  typedef typename ValueFunctionBase::State State;
  typedef typename ValueFunctionBase::StateBatch StateBatch;
  using RecurrentState = Eigen::VectorXd;
  using RecurrentStateBatch = Eigen::MatrixXd;
  typedef typename ValueFunctionBase::Value Value;
  typedef typename ValueFunctionBase::ValueBatch ValueBatch;
  typedef typename ValueFunctionBase::Gradient Gradient;

  RecurrentValueFunction_TensorFlow(std::string pathToGraphDefProtobuf, Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(
          pathToGraphDefProtobuf, learningRate) {
  }

  RecurrentValueFunction_TensorFlow(std::string computeMode,
                                    std::string graphName,
                                    std::string graphParam,
                                    Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(
          "RecurrentVfunction", computeMode, graphName, graphParam, learningRate) {
  }

  ~RecurrentValueFunction_TensorFlow() {};

  virtual void forward(State &state, RecurrentState &rcrntState, Dtype &value) {
    std::vector<MatrixXD> vectorOfOutputs;
    this->tf_->run({{"state", state},
                    {"init_rcrnt_state", rcrntState},
                    {"updateBNparams", this->notUpdateBN}},
                   {"value"}, {}, vectorOfOutputs);
    value = vectorOfOutputs[0](0);
  }

  virtual void forward(StateBatch &states, RecurrentStateBatch &rcrntStates, ValueBatch &values) {
    std::vector<MatrixXD> vectorOfOutputs;
    this->tf_->run({{"state", states},
                    {"init_rcrnt_state", rcrntStates}
                       { "updateBNparams", this->notUpdateBN }},
                   {"value"}, {}, vectorOfOutputs);
    values = vectorOfOutputs[0];
  }

  virtual Dtype performOneSolverIter(std::vector<StateBatch> &states,
                                     std::vector<RecurrentStateBatch> &rcrntStates,
                                     std::vector<ValueBatch> &values) {
    std::vector<MatrixXD> loss, dummy;
    this->tf_->run({{"state", states},
                    {"targetValue", values},
                    {"new_rcrnt_state", rcrntStates},
                    {"trainUsingTargetValue/learningRate", this->learningRate_},
                    {"updateBNparams", this->notUpdateBN}},
                   {"trainUsingTargetValue/loss"},
                   {"trainUsingTargetValue/solver"}, loss);
    return loss[0](0);
  }

 protected:
  using MatrixXD = typename TensorFlowNeuralNetwork<Dtype>::MatrixXD;

};
} // namespace FuncApprox
} // namespace RAI
