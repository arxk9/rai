//
// Created by joonho on 12/22/17.
//


#include <rai/function/tensorflow/StochasticPolicy_TensorFlow.hpp>
#include "rai/noiseModel/NormalDistributionNoise.hpp"
#include <rai/function/common/SimpleMLPLayer.hpp>
#include "rai/RAI_core"
#include "rai/RAI_Tensor.hpp"

using Dtype = float;
constexpr int StateDim = 13;
constexpr int ActionDim = 7;
using policy = rai::FuncApprox::StochasticPolicy_TensorFlow<Dtype, StateDim, ActionDim>;
typedef Eigen::Matrix<Dtype, ActionDim, 1> Action;
typedef Eigen::Matrix<Dtype, StateDim, 1> State;

int main() {

  RAI_init();

  std::string graphparam;
  graphparam = "relu 1e-3 13 64 32 32 7";
  policy network("cpu", "MLP", graphparam, 0.01);
  network.dumpParam(RAI_LOG_PATH + "/paramtest.txt");
  std::string filename;
  filename = RAI_LOG_PATH +"/paramtest.txt";
  rai::FuncApprox::MLP_fullyconnected<Dtype, StateDim, ActionDim> copynet(filename,"relu",{64, 32, 32});

  rai::Tensor<Dtype, 3> state("state");
  rai::Tensor<Dtype, 3> action("sampledAction");

  state.resize(StateDim, 1, 1);
  action.resize(ActionDim, 1, 1);

  state.setRandom();
  network.forward(state, action);

  State stateMat;
  Action actionMat, actionMat2;
  stateMat = state.batch(0);
  actionMat = action.batch(0);
  actionMat2 = copynet.forward(stateMat);
  std::cout << actionMat << std::endl << std::endl;
  std::cout <<actionMat2 << std::endl;

  std::cout <<"error sum: " << (actionMat - actionMat2).sum();


}
