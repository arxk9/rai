/*
 * master.cpp
 *
 *  Created on: Mar 7, 2016
 *      Author: jemin
 *
 *  Note"
 *	1. quadrotor task with AG tree (Hwangbo et. al. 2017)
 *	2. takes about an hour with gpu
 *
 */

#include "rai/RAI_core"

// Eigen
#include <Eigen/Dense>

// task
#include "rai/tasks/quadrotor/QuadrotorControl.hpp"

// noise model
#include "rai/noiseModel/NormalDistributionNoise.hpp"

// Neural network
#include "rai/function/tensorflow/DeterministicPolicy_TensorFlow.hpp"
#include "rai/function/tensorflow/ValueFunction_TensorFlow.hpp"

// algorithm
#include "rai/algorithm/AG_tree.hpp"

// acquisitor
#include "rai/experienceAcquisitor/TrajectoryAcquisitor_MultiThreadBatch.hpp"
#include "rai/experienceAcquisitor/TrajectoryAcquisitor_SingleThreadBatch.hpp"

using namespace std;
using namespace boost;

/// learning states
using Dtype = double;

/// shortcuts
using rai::Task::ActionDim;
using rai::Task::StateDim;
using rai::Task::CommandDim;
using Task_ = rai::Task::QuadrotorControl<Dtype>;
using NoiseCovariance = Eigen::Matrix<Dtype, ActionDim, ActionDim>;
using Policy = rai::FuncApprox::Policy_TensorFlow<Dtype, StateDim, ActionDim>;
using Vfunction = rai::FuncApprox::ValueFunction_TensorFlow<Dtype, StateDim>;
using Acquisitor = rai::ExpAcq::TrajectoryAcquisitor_SingleThreadBatch<Dtype, StateDim, ActionDim>;

int main(int argc, char *argv[]) {

  RAI_init();
  omp_set_num_threads(nThread);

  ////////////////////////// Define task ////////////////////////////
  Task_ task;
  task.setDiscountFactor(0.99);
  task.setControlUpdate_dt(0.01);
  task.setTimeLimitPerEpisode(5.0);

  ////////////////////////// Define Function approximations //////////
  Policy policy("policy_2l", "18 4 tanh 64 64 3e-3", 1e-3);
  Vfunction vfunction("Vfunction_2l", "18 tanh 64 64 3e-3", 1e-3);

  ////////////////////////// Define Noise Model //////////////////////
  rai::Noise::NormalDistributionNoise<Dtype, ActionDim> noise(NoiseCovariance::Identity() * Dtype(0.2));

  ////////////////////////// Acquisitor //////////////////////
  Acquisitor acquisitor;

  ////////////////////////// Algorithm ////////////////////////////////
  rai::Vector<rai::Task::Task<Dtype,StateDim,ActionDim,0> *> taskVector = {&task};
  rai::Vector<rai::Noise::NormalDistributionNoise<Dtype, ActionDim>*> noiseVector = {&noise};
  rai::Algorithm::AG_tree<Dtype, StateDim, ActionDim>
      algorithm(taskVector, &vfunction, &policy, noiseVector, &acquisitor, 512, 1024, 1, 1.5, 1.5, 350, 20);
  algorithm.setVisualizationLevel(1);

  /////////////////////// Plotting properties ////////////////////////
  rai::Utils::Graph::FigProp2D
      figurePropertiesEVP("N. Steps Taken", "Performance", "Number of Episodes vs Performance");

  rai::Utils::Graph::FigPropPieChart propChart;
  rai::Utils::logger->addVariableToLog(1, "process time", "");


  ////////////////////////// Learning /////////////////////////////////
  for (int iterationNumber = 0; iterationNumber < 300; iterationNumber++) {
    rai::Utils::logger->appendData("process time", rai::Utils::timer->getGlobalElapsedTimeInMin());
    LOG(INFO) << iterationNumber << "th loop";
    algorithm.runOneLoop();
    graph->figure(1, figurePropertiesEVP);
    graph->appendData(1, logger->getData("process time", 0),
                      logger->getData("Nominal performance", 1),
                      logger->getDataSize("Nominal performance"),
                      rai::Utils::Graph::PlotMethods2D::linespoints,
                      "performance",
                      "lw 2 lc 4 pi 1 pt 5 ps 1");
    graph->drawFigure(1, rai::Utils::Graph::OutputFormat::pdf);

    if(iterationNumber % 200 == 49) {
      policy.dumpParam(RAI_LOG_PATH + "/policy_" + std::to_string(iterationNumber) + ".txt");
      vfunction.dumpParam(RAI_LOG_PATH + "/value_" + std::to_string(iterationNumber) + ".txt");
    }
  }

  graph->drawPieChartWith_RAI_Timer(3, timer->getTimedItems(), propChart);
  graph->drawFigure(3, rai::Utils::Graph::OutputFormat::pdf);
}