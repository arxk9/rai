#include <rai/algorithm/common/LearningData.hpp>
#include "rai/function/common/ValueFunction.hpp"
#include "common/ParameterizedFunction_TensorFlow.hpp"

namespace rai {
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
  typedef typename Pfunction_tensorflow ::InnerState InnerState;


  RecurrentValueFunction_TensorFlow(std::string pathToGraphDefProtobuf, Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(
          pathToGraphDefProtobuf, learningRate) {
  }

  RecurrentValueFunction_TensorFlow(std::string computeMode,
                                    std::string graphName,
                                    std::string graphParam,
                                    Dtype learningRate = 1e-3) :
      Pfunction_tensorflow::ParameterizedFunction_TensorFlow(
          "RecurrentVfunction", computeMode, graphName, graphParam, learningRate), h("h_init") {
    hdim = this->getInnerStatesize();
    h.resize(hdim, 0);
    h.setZero();
  }

  ~RecurrentValueFunction_TensorFlow() {};

  virtual void forward(State &state, Dtype &value) {
    std::vector<MatrixXD> vectorOfOutputs;
    MatrixXD h_, length;
    length.resize(1,1);
    h_.resize(hdim,1);
    h_.setZero();

    this->tf_->run({{"state", state},
                    {"length", length},
                    {"h_init", h_}},
                   {"value"}, {}, vectorOfOutputs);
    value = vectorOfOutputs[0](0);
  }
  virtual void forward(StateBatch &states, ValueBatch &values) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    Tensor3D stateT({stateDim, 1, states.cols()}, "state");
    Tensor1D len({states.cols()}, 1, "length");
    stateT.copyDataFrom(states);

    if (h.cols() != states.cols()) {
      h.resize(hdim, states.cols());
    }
    h.setZero();
//    }

    this->tf_->run({stateT,  h, len}, {"value", "h_state"}, {}, vectorOfOutputs);
    std::memcpy(values.data(), vectorOfOutputs[0].flat<Dtype>().data(), sizeof(Dtype) * values.size());
    h.copyDataFrom(vectorOfOutputs[1]);
  }

  virtual void forward(Tensor3D &states, Tensor3D &values) {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    Tensor1D len({states.batches()}, states.dim(1), "length");

    if (h.cols() != states.batches()) {
      h.resize(hdim, states.batches());
    }
    h.setZero();
//    }

    this->tf_->run({states,  h, len}, {"value", "h_state"}, {}, vectorOfOutputs);
//    h.copyDataFrom(vectorOfOutputs[1]);
    values.copyDataFrom(vectorOfOutputs[0]);
//    LOG(INFO) << h.eMat();
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
//  virtual Dtype performOneSolverIter(Dataset *minibatch, Tensor3D &values){
//    std::vector<MatrixXD> vectorOfOutputs;
//    values = "targetValue";
//    Tensor1D lr({1}, this->learningRate_(0), "trainUsingTargetValue/learningRate");
//
//    if(h.cols()!= minibatch->batchNum) h.resize(hdim, minibatch->batchNum);
//    h.setZero();
//
//    this->tf_->run({minibatch->states, minibatch->lengths, values, h, lr},
//                   {"trainUsingTargetValue/loss"},
//                   {"trainUsingTargetValue/solver"}, vectorOfOutputs);
//
//    return vectorOfOutputs[0](0);
//  };

  virtual Dtype performOneSolverIter_trustregion(StateBatch &states, ValueBatch &values, ValueBatch &old_values) {
    std::vector<MatrixXD> loss, dummy;
    this->tf_->run({{"state", states},
                    {"targetValue", values},
                    {"predictedValue", old_values},
                    {"trainUsingTRValue/learningRate", this->learningRate_},
                    {"updateBNparams", this->notUpdateBN}},
                   {"trainUsingTRValue/loss"},
                   {"trainUsingTRValue/solver"}, loss);
    return loss[0](0);
  }

//  template <int actionDim>
//  virtual Dtype performOneSolverIter_trustregion(rai::Algorithm::LearningData<Dtype,stateDim, actionDim>* minibatch , Tensor2D &old_values) {
//    std::vector<MatrixXD> loss;
//    Tensor1D lr({1}, this->learningRate_(0), "trainUsingTRValue/learningRate");
//    Tensor2D hiddenState({hiddenStateDim(),  minibatch->states.batches()},0, "h_init");
//
//    this->tf_->run({minibatch->states,
//                    minibatch->values,
//                    minibatch->lengths,
//                    old_values,
//                    lr,hiddenState},
//                   {"trainUsingTRValue/loss"},
//                   {"trainUsingTRValue/solver"}, loss);
//    return loss[0](0);
//  }

  virtual int getInnerStatesize() {
    std::vector<tensorflow::Tensor> vectorOfOutputs;
    this->tf_->run({}, {"h_dim"}, {}, vectorOfOutputs);
    return vectorOfOutputs[0].scalar<int>()();
  }

  virtual void getInnerStates(InnerState &h_out){
    h_out = h;
  }

  int hiddenStateDim() { return hdim; }

 protected:
  using MatrixXD = typename TensorFlowNeuralNetwork<Dtype>::MatrixXD;
  int hdim = 0;
  Tensor2D h;
};
} // namespace FuncApprox
} // namespace rai
