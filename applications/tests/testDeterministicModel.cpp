//
// Created by jhwangbo on 01/09/17.
//

//
// Created by jhwangbo on 01/09/17.
//

#include "rai/function/tensorflow/DeterministicModel_TensorFlow.hpp"
#include <fstream>
#include <iterator>
#include <vector>
#include "rai/RAI_core"
#include "rai/RAI_Tensor.hpp"
#include <immintrin.h>

int main() {

  RAI_init();
  RAI::Tensor<double, 1> l1coefT({1}, "L1_loss_coef"), l2coefT({1}, "L2_loss_coef"), learningRate({1}, "squareLoss/learningRate"), loss;
  l1coefT.data()[0] = 0.1;
  l2coefT.data()[0] = 0.1;
  learningRate.data()[0] = 0.001;
  RAI::Tensor<double, 2> inputT({24,575916}, "input"), targetT({1,575916}, "targetOutput");
  RAI::FuncApprox::DeterministicModel_TensorFlow<double, 24, 8>
      sea("cpu", "MLP_inputBottleneck", "relu 1e-3 24 64 64 64 1", 0.001);

  std::ifstream input("/home/jhwangbo/Documents/data_input.bin", std::ios::in | std::ios::binary);
  std::ifstream target("/home/jhwangbo/Documents/data_target.bin", std::ios::in | std::ios::binary);

  // copies all data into buffer
  input.read((char*)inputT.data(), 24*575916*sizeof(double));
  input.close();

  target.read((char*)targetT.data(), 575916*sizeof(double));
  target.close();
  std::vector<tensorflow::Tensor> vectorOfOutputs;

  RAI::Utils::Graph::FigProp2D prop("time", "torque", "model");
  prop.size = "1200,800";
  Eigen::VectorXd time = Eigen::VectorXd::LinSpaced(10000, 0, 25);

  for(int i=0; i<100; i++) {
    sea.run({l1coefT, l2coefT, inputT, targetT, learningRate},
            {"output", "squareLoss/loss", "L1_loss", "L2_loss"},
            {"squareLoss/solver"},
            loss.output());
    std::cout << "loss " << loss[1][0] << "   L1_loss "<< loss[2][0] << "   L2_loss "<< loss[3][0] << std::endl;
    RAI::Utils::graph->figure(0, prop);
    RAI::Utils::graph->appendData(0, time.data(), loss[0], 10000, "torque");
    RAI::Utils::graph->appendData(0, time.data(), targetT.data(), 10000, "target");
    RAI::Utils::graph->drawFigure(0);
  }
}