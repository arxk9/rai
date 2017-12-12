#include <rai/algorithm/common/LearningData.hpp>
#include "rai/function/common/ValueFunction.hpp"
#include "common/RecurrentParametrizedFunction_TensorFlow.hpp"

namespace rai {
namespace FuncApprox {

template<typename Dtype, int stateDim>
class RecurrentValueFunction_TensorFlow : public virtual ValueFunction<Dtype, stateDim>,
                                          public virtual RecurrentParameterizedFunction_TensorFlow<Dtype,
                                                                                                   stateDim,
                                                                                                   1>{

 public:
  typedef Eigen::Matrix<Dtype, -1, 1> VectorXD;
  typedef Eigen::Matrix<Dtype, -1, -1> MatrixXD;
  using ValueFunctionBase = ValueFunction<Dtype, stateDim>;
  using Pfunction_tensorflow = RecurrentParameterizedFunction_TensorFlow<Dtype, stateDim, 1>;

  using Pfunction_tensorflow::h;
  using Pfunction_tensorflow::hdim;

  typedef typename ValueFunctionBase::State State;
  typedef typename ValueFunctionBase::StateBatch StateBatch;
  typedef typename ValueFunctionBase::Value Value;
  typedef typename ValueFunctionBase::ValueBatch ValueBatch;
  typedef typename ValueFunctionBase::Gradient Gradient;

  typedef typename ValueFunctionBase::Tensor1D Tensor1D;
  typedef typename ValueFunctionBase::Tensor2D Tensor2D;
  typedef typename ValueFunctionBase::Tensor3D Tensor3D;

  RecurrentValueFunction_TensorFlow(std::string pathToGraphDefProtobuf, Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::RecurrentParameterizedFunction_TensorFlow(
          pathToGraphDefProtobuf, learningRate) {
  }

  RecurrentValueFunction_TensorFlow(std::string computeMode,
                                    std::string graphName,
                                    std::string graphParam,
                                    Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::RecurrentParameterizedFunction_TensorFlow(
          "RecurrentVfunction", computeMode, graphName, graphParam, learningRate){
  }

  ~RecurrentValueFunction_TensorFlow() {};

  virtual void test(Tensor3D &states, Tensor2D &values) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    Tensor1D len({states.batches()}, states.dim(1), "length");

    LOG(INFO) << len.eMat().transpose() << std::endl;
    if (h.cols() != states.batches()) {
      h.resize(hdim, states.batches());
    }
    h.setZero();

    this->tf_->run({states,  h, len}, {"test", "h_state"}, {}, vectorOfOutputs);
//    h.copyDataFrom(vectorOfOutputs[1]);
    values.copyDataFrom(vectorOfOutputs[0]);
  }


  virtual Dtype performOneSolverIter(Tensor3D &states, Tensor2D &values, Tensor1D & lengths) {
    std::vector<MatrixXD> loss;
    Tensor1D lr({1}, this->learningRate_, "trainUsingTargetValue/learningRate");
    Tensor2D hiddenState({hiddenStateDim(), states.batches()},0, "h_init");

    this->tf_->run({states,
                    values,
                    lengths,
                    hiddenState,
                    lr},
                   {"trainUsingTargetValue/loss"},
                   {"trainUsingTargetValue/solver"}, loss);
    return loss[0](0);
  }

  virtual Dtype performOneSolverIter_trustregion(Tensor3D &states, Tensor2D &values, Tensor2D &old_values, Tensor1D & lengths) {
    std::vector<MatrixXD> loss;
    Tensor1D lr({1}, this->learningRate_, "trainUsingTRValue/learningRate");
    Tensor2D hiddenState({hiddenStateDim(), states.batches()},0, "h_init");

    this->tf_->run({states,
                    values,
                    old_values,
                    lengths,
                    hiddenState,
                    lr},
                   {"trainUsingTRValue/loss"},
                   {"trainUsingTRValue/solver"}, loss);
    return loss[0](0);
  }
};
} // namespace FuncApprox
} // namespace rai
