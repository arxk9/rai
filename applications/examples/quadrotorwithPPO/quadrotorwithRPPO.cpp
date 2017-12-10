//
// Created by jhwangbo on 10/08/17.
//


#include <rai/RAI_core>

// Eigen
#include <Eigen/Dense>

// task
#include "rai/tasks/quadrotor/QuadrotorControl.hpp"

// noise model
#include "rai/noiseModel/NormalDistributionNoise.hpp"

// Neural network
#include "rai/function/tensorflow/RecurrentStochasticPolicyValue_TensorFlow.hpp"

// algorithm
#include <rai/experienceAcquisitor/TrajectoryAcquisitor_Parallel.hpp>
#include "rai/algorithm/RPPO.hpp"

using namespace std;
using namespace boost;

/// learning states
using Dtype = double;

/// shortcuts
using rai::Task::ActionDim;
using rai::Task::StateDim;
using rai::Task::CommandDim;
using Task = rai::Task::QuadrotorControl<Dtype>;

using State = Task::State;
using StateBatch = Task::StateBatch;
using Action = Task::Action;
using ActionBatch = Task::ActionBatch;
using CostBatch = Task::CostBatch;
using VectorXD = Task::VectorXD;
using MatrixXD = Task::MatrixXD;
using PolicyValue_TensorFlow = rai::FuncApprox::RecurrentStochasticPolicyValue_Tensorflow<Dtype, StateDim, ActionDim>;

using Acquisitor_ = rai::ExpAcq::TrajectoryAcquisitor_Parallel<Dtype, StateDim, ActionDim>;
using Noise = rai::Noise::NormalDistributionNoise<Dtype, ActionDim>;
using NoiseCovariance = Eigen::Matrix<Dtype, ActionDim, ActionDim>;

#define nThread 10

int main(int argc, char *argv[]) {

  RAI_init();
  omp_set_num_threads(nThread);

  ////////////////////////// Define task ////////////////////////////
  std::vector<Task> taskVec(nThread, Task());
  std::vector<rai::Task::Task<Dtype, StateDim, ActionDim, 0> *> taskVector;

  for (auto &task : taskVec) {
    task.setControlUpdate_dt(0.01);
    task.setDiscountFactor(0.99);
    task.setTimeLimitPerEpisode(5.0);
    taskVector.push_back(&task);
  }

  ////////////////////////// Define Noise Model //////////////////////
  NoiseCovariance covariance = NoiseCovariance::Identity() ;
  std::vector<Noise> noiseVec(nThread, Noise(covariance));
  std::vector<Noise *> noiseVector;
  for (auto &noise : noiseVec)
    noiseVector.push_back(&noise);

  ////////////////////////// Define Function approximations //////////
  PolicyValue_TensorFlow policy("gpu,0", "LSTM_merged", "relu 1e-3 18 128 / 128 64 4", 0.0025);

  ////////////////////////// Acquisitor
  Acquisitor_ acquisitor;

  ////////////////////////// Algorithm ////////////////////////////////
  rai::Algorithm::RPPO<Dtype, StateDim, ActionDim>
      algorithm(taskVector,&policy, noiseVector, &acquisitor, 0.95, 2, 10, 10, 4, 4, 50, 1, true, 0.3);

  algorithm.setVisualizationLevel(0);

  /////////////////////// Plotting properties ////////////////////////
  rai::Utils::Graph::FigProp2D figurePropertiesEVP;
  figurePropertiesEVP.title = "Number of Steps Taken vs Performance";
  figurePropertiesEVP.xlabel = "N. Steps Taken";
  figurePropertiesEVP.ylabel = "Performance";

  rai::Utils::Graph::FigProp2D figurePropertieskl;
  figurePropertieskl.title = "Number of Steps Taken vs KlD";
  figurePropertieskl.xlabel = "N. Steps Taken";
  figurePropertieskl.ylabel = "KlD";

  rai::Utils::Graph::FigProp2D figurePropertiesgnorm;
  figurePropertieskl.title = "Number of Steps Taken vs gradNorm";
  figurePropertieskl.xlabel = "N. Steps Taken";
  figurePropertieskl.ylabel = "gradNorm";

  rai::Utils::Graph::FigPropPieChart propChart;

  ////////////////////////// Learning /////////////////////////////////
  constexpr int loggingInterval = 20;
  int iteration = 5000;
  Dtype lr = policy.getLearningRate();
  Dtype lr_lowerbound =  0.1 * lr;

  for (int iterationNumber = 0; iterationNumber < iteration; iterationNumber++) {
    LOG(INFO) << iterationNumber << "th Iteration";

    lr = (1-(Dtype)iterationNumber/iteration) * (lr-lr_lowerbound) + lr_lowerbound; ///linear decay
    policy.setLearningRate(lr);

    if (iterationNumber % loggingInterval == 0) {
      algorithm.setVisualizationLevel(1);
      taskVector[0]->enableVideoRecording();
    }
    LOG(INFO) << "Learning rate:"<<lr;

    algorithm.runOneLoop(20000);

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
      graph->drawFigure(1, rai::Utils::Graph::OutputFormat::pdf);


      graph->figure(2, figurePropertieskl);
      graph->appendData(2, logger->getData("klD", 0),
                        logger->getData("klD", 1),
                        logger->getDataSize("klD"),
                        rai::Utils::Graph::PlotMethods2D::linespoints,
                        "klD",
                        "lw 2 lc 4 pi 1 pt 5 ps 1");
      graph->drawFigure(2, rai::Utils::Graph::OutputFormat::pdf);

      graph->figure(3, figurePropertiesgnorm);
      graph->appendData(3, logger->getData("gradnorm", 0),
                        logger->getData("gradnorm", 1),
                        logger->getDataSize("gradnorm"),
                        rai::Utils::Graph::PlotMethods2D::linespoints,
                        "gradnorm",
                        "lw 2 lc 4 pi 1 pt 5 ps 1");
      graph->drawFigure(3, rai::Utils::Graph::OutputFormat::pdf);

    }

  }

  policy.dumpParam(RAI_LOG_PATH + "/policy.txt");
  graph->drawPieChartWith_RAI_Timer(0, timer->getTimedItems(), propChart);
  graph->drawFigure(0, rai::Utils::Graph::OutputFormat::pdf);
  graph->waitForEnter();

}