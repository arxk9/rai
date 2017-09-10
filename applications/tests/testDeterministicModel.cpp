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
  RAI::Tensor<double, 1> learningRate({1}, "squareLoss/learningRate"), loss;
  learningRate.data()[0] = 0.0005;
  RAI::Tensor<double, 2> inputT({9, 575916}, "input"), targetT({1,575916}, "targetOutput");

  for(int i=1; i < 9 ; i++)
    inputT.row(i) /= 8.0;

  RAI::FuncApprox::DeterministicModel_TensorFlow<double, 9, 1>
      sea("cpu", "MLP_inputBottleneck", "relu 1e-3 9 64 64 64 64 1", 0.001);

  std::ifstream input("/home/jhwangbo/Documents/data_input5.bin", std::ios::in | std::ios::binary);
  std::ifstream target("/home/jhwangbo/Documents/data_target5.bin", std::ios::in | std::ios::binary);

  // copies all data into buffer
  input.read((char*)inputT.data(), 9*575916*sizeof(double));
  input.close();

  target.read((char*)targetT.data(), 575916*sizeof(double));
  target.close();
  std::vector<tensorflow::Tensor> vectorOfOutputs;

  RAI::Utils::Graph::FigProp2D prop("time", "torque", "model");
  prop.size = "1200,800";
  Eigen::VectorXd time = Eigen::VectorXd::LinSpaced(2000, 0, 5);
  Eigen::VectorXd command = inputT.row(0);

  for(int i=0; i<1000; i++) {
    sea.run({inputT, targetT, learningRate},
            {"output", "squareLoss/loss"},
            {"squareLoss/solver"},
            loss.output());
    std::cout << "loss " << loss[1][0] << std::endl;
    RAI::Utils::graph->figure(0, prop);
    RAI::Utils::graph->appendData(0, time.data(), loss[0]+100000, 2000, "torque");
    RAI::Utils::graph->appendData(0, time.data(), targetT.data()+100000, 2000, "target");
    RAI::Utils::graph->appendData(0, time.data(), command.data()+100000, 2000, "command");

    if(i==999)
      RAI::Utils::graph->drawFigure(0, RAI::Utils::Graph::OutputFormat::pdf);
    else
      RAI::Utils::graph->drawFigure(0);
  }
  sea.dumpParam("seaModel");
  RAI::Utils::graph->waitForEnter();
}