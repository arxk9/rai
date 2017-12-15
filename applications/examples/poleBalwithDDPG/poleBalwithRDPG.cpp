//
// Created by joonho on 12/9/17.
//

#include <rai/RAI_core>

// Eigen
#include <Eigen/Dense>

// task
//#include "rai/tasks/poleBalancing/PoleBalancing.hpp"
#include "rai/tasks/poleBalancing/POPoleBalancing.hpp"

// noise model
#include "rai/noiseModel/OrnsteinUhlenbeckNoise.hpp"

// Neural network
#include "rai/function/tensorflow/RecurrentQfunction_TensorFlow.hpp"
#include "rai/function/tensorflow/RecurrentDeterministicPolicy_Tensorflow.hpp"

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
//using Task = rai::Task::PoleBalancing<Dtype>;
using Task = rai::Task::PO_PoleBalancing<Dtype>;

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
using Acquisitor_ = rai::ExpAcq::TrajectoryAcquisitor_Parallel<Dtype, StateDim, ActionDim>;
using Noise = rai::Noise::Noise<Dtype, ActionDim>;

using NormalNoise = rai::Noise::NormalDistributionNoise<Dtype, ActionDim>;
using NoiseCovariance = Eigen::Matrix<Dtype, ActionDim, ActionDim>;


Dtype learningRateQfunction = 0.001;
Dtype learningRatePolicy = 0.001;
#define nThread 10

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
    task.setTimeLimitPerEpisode(25.0);
    taskVector.push_back(&task);
  }

  ////////////////////////// Define Noise Model //////////////////////
  std::vector<rai::Noise::OrnsteinUhlenbeck<Dtype, ActionDim>>
      noiseVec(nThread, rai::Noise::OrnsteinUhlenbeck<Dtype, ActionDim>(0.15, 0.3));
  std::vector<Noise *> noiseVector;
  for (auto &noise : noiseVec)
    noiseVector.push_back(&noise);
//
//  NoiseCovariance covariance = NoiseCovariance::Identity();
//  std::vector<NormalNoise> noiseVec(nThread, NormalNoise(covariance));
//  std::vector<Noise *> noiseVector;
//  for (auto &noise : noiseVec)
//    noiseVector.push_back(&noise);

  ////////////////////////// Define Memory ////////////////////////////
  ReplayMemory Memory(100);

  ////////////////////////// Define Function approximations //////////
  Policy_TensorFlow policy("gpu,0", "LSTMMLP", "relu 1e-3 2 64 / 32 1", learningRatePolicy);
  Policy_TensorFlow policy_target("gpu,0", "LSTMMLP", "relu 1e-3 2 64 / 32 1", learningRatePolicy);

  Qfunction_TensorFlow qfunction("gpu,0", "LSTMMLP2", "relu 1e-3 2 1 64 / 32 1", learningRateQfunction);
  Qfunction_TensorFlow qfunction_target("gpu,0", "LSTMMLP2", "relu 1e-3 2 1 64 / 32 1", learningRateQfunction);


  ////////////////////////// Acquisitor
  Acquisitor_ acquisitor;

  ////////////////////////// Algorithm and Hyperparameters /////////////////////////
  rai::Algorithm::RDPG<Dtype, StateDim, ActionDim>
      algorithm(taskVector,
                &qfunction,
                &qfunction_target,
                &policy,
                &policy_target,
                noiseVector,
                &acquisitor,
                &Memory,
                1,
                5,
                5,
                25,
                50, 1);
  algorithm.setVisualizationLevel(0);
  algorithm.initiallyFillTheMemory();

  policy.setMaxGradientNorm(0.25);
  policy.setLearningRateDecay(0.99,10);
  qfunction.setLearningRateDecay(0.99,10);
  qfunction.setMaxGradientNorm(0.25);

  /////////////////////// Plotting properties ////////////////////////
  rai::Utils::Graph::FigProp2D
      figurePropertiesEVP("N. Steps Taken", "Performance", "Number of Steps Taken vs Performance");

  rai::Utils::Graph::FigProp2D figurePropertiesgnorm;
  figurePropertiesgnorm.title = "Number of Steps Taken vs gradNorm";
  figurePropertiesgnorm.xlabel = "N. Steps Taken";
  figurePropertiesgnorm.ylabel = "gradNorm";

  rai::Utils::Graph::FigPropPieChart propChart;

  constexpr int loggingInterval = 10;

  ////////////////////////// Learning /////////////////////////////////
  for (int iterationNumber = 0; iterationNumber < 101; iterationNumber++) {
    LOG(INFO) << iterationNumber << "th Iteration";
    LOG(INFO) << "Learning rate:"<<policy.getLearningRate();
    LOG(INFO) << "Number of updates:"<<policy.getGlobalStep();

    if (iterationNumber % loggingInterval == 0) {
      algorithm.setVisualizationLevel(1);
      taskVector[0]->enableVideoRecording();
    }

    algorithm.learnForNepisodes(10);
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

      graph->figure(2, figurePropertiesgnorm);
      graph->appendData(2, logger->getData("gradnorm", 0),
                        logger->getData("gradnorm", 1),
                        logger->getDataSize("gradnorm"),
                        rai::Utils::Graph::PlotMethods2D::linespoints,
                        "gradnorm",
                        "lw 2 lc 4 pi 1 pt 5 ps 1");
      graph->drawFigure(2, rai::Utils::Graph::OutputFormat::pdf);

    }
  }
  policy.dumpParam(RAI_LOG_PATH + "/policy.txt");
  graph->drawPieChartWith_RAI_Timer(5, timer->getTimedItems(), propChart);
  graph->drawFigure(5, rai::Utils::Graph::OutputFormat::pdf);
  graph->waitForEnter();

}