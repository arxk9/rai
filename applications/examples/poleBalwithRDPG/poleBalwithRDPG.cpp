//
// Created by jhwangbo on 22.09.16.
//

/*
 * master.cpp
 *
 *  Created on: Mar 7, 2016
 *      Author: jemin
 *
 *
 *											   generalized coordinates
 *  Note"
 *	1. Visualize with a single CPU only
 *
 */

#include <rai/RAI_core>

// Eigen
#include <Eigen/Dense>

// task
#include "rai/tasks/poleBalancing/PoleBalancing.hpp"
//#include "rai/tasks/poleBalancing/POPoleBalancing.hpp"

// noise model
#include "rai/noiseModel/OrnsteinUhlenbeckNoise.hpp"

// Neural network
#include "rai/function/tensorflow/Qfunction_TensorFlow.hpp"
#include "rai/function/tensorflow/DeterministicPolicy_TensorFlow.hpp"

// algorithm
#include "rai/algorithm/RDPG.hpp"

// acquisitor
#include <rai/experienceAcquisitor/ExperienceTupleAcquisitor_Sequential.hpp>
#include <rai/experienceAcquisitor/ExperienceTupleAcquisitor_Parallel.hpp>

using namespace std;
using namespace boost;

/// learning states
using Dtype = float;

/// shortcuts
using rai::Task::ActionDim;
using rai::Task::StateDim;
using rai::Task::CommandDim;
using Task = rai::Task::PoleBalancing<Dtype>;
//using Task = rai::Task::PO_PoleBalancing<Dtype>;

using State = Task::State;
using StateBatch = Task::StateBatch;
using Action = Task::Action;
using ActionBatch = Task::ActionBatch;
using CostBatch = Task::CostBatch;
using VectorXD = Task::VectorXD;
using MatrixXD = Task::MatrixXD;

using Policy_TensorFlow = rai::FuncApprox::RecurrentDeterministicPolicy_TensorFlow<Dtype, StateDim, ActionDim>;
using Qfunction_TensorFlow = rai::FuncApprox::RecurrentQfunction_TensorFlow<Dtype, StateDim, ActionDim>;
using ReplayMemory = rai::Memory::ReplayMemoryHistory<Dtype, StateDim, ActionDim>;
using Acquisitor_ = rai::ExpAcq::TrajectoryAcquisitor_MultiThreadBatch<Dtype, StateDim, ActionDim>;
using Noise = rai::Noise::Noise<Dtype, ActionDim>;

using NormalNoise = rai::Noise::NormalDistributionNoise<Dtype, ActionDim>;
using NoiseCovariance = Eigen::Matrix<Dtype, ActionDim, ActionDim>;


Dtype learningRateQfunction = 1e-3;
Dtype learningRatePolicy = 1e-3;
#define nThread 4

int main(int argc, char *argv[]) {

  RAI_init();
  omp_set_num_threads(nThread);

  ////////////////////////// Define task ////////////////////////////
  std::vector<Task> taskVec(nThread, Task(Task::fixed, Task::easy));
  std::vector<rai::Task::Task<Dtype, StateDim, ActionDim, 0> *> taskVector;

  for (auto &task : taskVec) {
    task.setControlUpdate_dt(0.05);
    task.setDiscountFactor(0.99);
    task.setRealTimeFactor(5);
    task.setTimeLimitPerEpisode(10.0);
    taskVector.push_back(&task);
  }

  ////////////////////////// Define Noise Model //////////////////////
//  std::vector<rai::Noise::OrnsteinUhlenbeck<Dtype, ActionDim>>
//      noiseVec(nThread, rai::Noise::OrnsteinUhlenbeck<Dtype, ActionDim>(0.2 , 1, 0.05));
//  std::vector<Noise *> noiseVector;
//  for (auto &noise : noiseVec)
//    noiseVector.push_back(&noise);

  NoiseCovariance covariance = NoiseCovariance::Identity();
  std::vector<NormalNoise> noiseVec(nThread, NormalNoise(covariance));
  std::vector<Noise *> noiseVector;
  for (auto &noise : noiseVec)
    noiseVector.push_back(&noise);

  ////////////////////////// Define Memory ////////////////////////////
  ReplayMemory Memory(1000);

  ////////////////////////// Define Function approximations //////////
  Policy_TensorFlow policy("gpu,0", "LSTMMLP", "tanh 1e-3 3 32 / 32 1", learningRatePolicy);
  Policy_TensorFlow policy_target("gpu,0", "LSTMMLP", "tanh 1e-3 3 64 / 32 1", learningRatePolicy);

  Qfunction_TensorFlow qfunction("gpu,0", "LSTMMLP2", "tanh 1e-3 3 1 64 / 32 1", learningRateQfunction);
  Qfunction_TensorFlow qfunction_target("gpu,0", "LSTMMLP2", "tanh 1e-3 3 1 64 / 32 1", learningRateQfunction);
//  rai::FuncApprox::Qfunction_TensorFlow<Dtype, StateDim, ActionDim> qfunction2("cpu", "MLP2", "relu 1e-3 3 1 32 32 1", learningRateQfunction);

  ////////////////////////// Acquisitor
  Acquisitor_ acquisitor;

  ////////////////////////// Algorithm ////////////////////////////////
  rai::Algorithm::RDPG<Dtype, StateDim, ActionDim>
      algorithm(taskVector,
                &qfunction,
                &qfunction_target,
                &policy,
                &policy_target,
                noiseVector,
                &acquisitor,
                &Memory,
                30,
                1);
  algorithm.setVisualizationLevel(0);
  algorithm.initiallyFillTheMemory();

  /////////////////////// Plotting properties ////////////////////////
  rai::Utils::Graph::FigProp2D
      figurePropertiesEVP("N. Steps Taken", "Performance", "Number of Steps Taken vs Performance");
  rai::Utils::Graph::FigProp3D figurePropertiesSVC("angle", "angular velocity", "value", "Q function");
  figurePropertiesSVC.displayType = rai::Utils::Graph::DisplayType3D::heatMap3D;
  rai::Utils::Graph::FigProp3D figurePropertiesSVA("angle", "angular velocity", "action", "Policy");
  figurePropertiesSVA.displayType = rai::Utils::Graph::DisplayType3D::heatMap3D;
  rai::Utils::Graph::FigProp3D
      figurePropertiesSVGradient("angle", "angular velocity", "value", "Qfunction training data");
  rai::Utils::Graph::FigPropPieChart propChart;

  StateBatch state_plot(3, 2601);
  ActionBatch action_plot(1, 2601);
  CostBatch value_plot(1, 2601);
  MatrixXD minimal_X_extended(1, 2601);
  MatrixXD minimal_Y_extended(1, 2601);

  MatrixXD minimal_X_sampled(1, 2601);
  MatrixXD minimal_Y_sampled(1, 2601);
  ActionBatch action_sampled(1, 2601);

  for (int i = 0; i < 51; i++) {
    for (int j = 0; j < 51; j++) {
      minimal_X_extended(i * 51 + j) = -M_PI + M_PI * i / 25.0;
      minimal_Y_extended(i * 51 + j) = -5.0 + j / 25.0 * 5.0;
      state_plot(0, i * 51 + j) = cos(minimal_X_extended(i * 51 + j));
      state_plot(1, i * 51 + j) = sin(minimal_X_extended(i * 51 + j));
      state_plot(2, i * 51 + j) = minimal_Y_extended(i * 51 + j);
    }
  }
  constexpr int loggingInterval = 10;

  ////////////////////////// Learning /////////////////////////////////
  for (int iterationNumber = 0; iterationNumber < 50; iterationNumber++) {

    if (iterationNumber % loggingInterval == 0) {
      algorithm.setVisualizationLevel(1);
      taskVector[0]->enableVideoRecording();
    }
    algorithm.learnForNepisodes(5);
    if (iterationNumber % loggingInterval == 0) {
      algorithm.setVisualizationLevel(0);
      taskVector[0]->disableRecording();

      graph->figure(1, figurePropertiesEVP);
      graph->appendData(1, logger->getData("PerformanceTester/performance", 0),
                        logger->getData("PerformanceTester/performance", 1),
                        logger->getDataSize("PerformanceTester/performance"),
                        rai::Utils::Graph::PlotMethods2D::linespoints,
                        "performance",
                        "lw 2 lc 4 pi 1 pt 5 ps 1");
      graph->drawFigure(1);

      policy.forward(state_plot, action_plot);
      qfunction.forward(state_plot, action_plot, value_plot);

      graph->drawHeatMap(3, figurePropertiesSVC, minimal_X_extended.data(),
                         minimal_Y_extended.data(), value_plot.data(), 51, 51, "");
      graph->drawFigure(3);
      graph->drawHeatMap(4, figurePropertiesSVA, minimal_X_extended.data(),
                         minimal_Y_extended.data(), action_plot.data(), 51, 51, "");
      graph->drawFigure(4);
          }
  }
  policy.dumpParam(RAI_LOG_PATH + "/policy.txt");
  graph->drawPieChartWith_RAI_Timer(5, timer->getTimedItems(), propChart);
  graph->drawFigure(5, rai::Utils::Graph::OutputFormat::pdf);
  graph->waitForEnter();

}