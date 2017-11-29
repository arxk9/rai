//
// Created by jhwangbo on 10/08/17.
//


#include <rai/RAI_core>

// Eigen
#include <Eigen/Dense>

// task
#include "rai/tasks/poleBalancing/POPoleBalancing.hpp"
//#include "rai/tasks/poleBalancing/PoleBalancing.hpp"

// noise model
#include "rai/noiseModel/NormalDistributionNoise.hpp"

// Neural network
#include "rai/function/tensorflow/RecurrentValueFunction_TensorFlow.hpp"
#include "rai/function/tensorflow/RecurrentStochasticPolicy_TensorFlow.hpp"

// algorithm
#include <rai/experienceAcquisitor/TrajectoryAcquisitor_Parallel.hpp>
#include "rai/algorithm/PPO.hpp"

using namespace std;
using namespace boost;

/// learning states
using Dtype = float;

/// shortcuts
using rai::Task::ActionDim;
using rai::Task::StateDim;
using rai::Task::CommandDim;
using Task = rai::Task::PO_PoleBalancing<Dtype>;
//using Task = rai::Task::PoleBalancing<Dtype>;

using State = Task::State;
using StateBatch = Task::StateBatch;
using Action = Task::Action;
using ActionBatch = Task::ActionBatch;
using CostBatch = Task::CostBatch;
using VectorXD = Task::VectorXD;
using MatrixXD = Task::MatrixXD;
using Policy_TensorFlow = rai::FuncApprox::RecurrentStochasticPolicy_TensorFlow<Dtype, StateDim, ActionDim>;
using Vfunction_TensorFlow = rai::FuncApprox::RecurrentValueFunction_TensorFlow<Dtype, StateDim>;
using Acquisitor_ = rai::ExpAcq::TrajectoryAcquisitor_Parallel<Dtype, StateDim, ActionDim>;
using Noise = rai::Noise::NormalDistributionNoise<Dtype, ActionDim>;
using NoiseCovariance = Eigen::Matrix<Dtype, ActionDim, ActionDim>;

#define nThread 10

int main(int argc, char *argv[]) {

  RAI_init();
  omp_set_num_threads(nThread);

  ////////////////////////// Define task ////////////////////////////
  std::vector<Task> taskVec(nThread, Task(Task::fixed, Task::easy));
  std::vector<rai::Task::Task<Dtype, StateDim, ActionDim, 0> *> taskVector;

  for (auto &task : taskVec) {
    task.setControlUpdate_dt(0.05);
    task.setDiscountFactor(0.995);
    task.setRealTimeFactor(2);
    task.setTimeLimitPerEpisode(10);
    taskVector.push_back(&task);
  }

  ////////////////////////// Define Noise Model //////////////////////
  Dtype Stdev = 1;

  NoiseCovariance covariance = NoiseCovariance::Identity() * Stdev;
  std::vector<Noise> noiseVec(nThread, Noise(covariance));
  std::vector<Noise *> noiseVector;
  for (auto &noise : noiseVec)
    noiseVector.push_back(&noise);

  ////////////////////////// Define Function approximations //////////
  Vfunction_TensorFlow Vfunction("gpu,0", "GRUMLP", "relu 1e-3 2 64 / 32 1", 0.001);
  Policy_TensorFlow policy("gpu,0", "GRUMLP", "tanh 1e-3 2 64 / 32 1", 0.001);
//  Policy_TensorFlow policy("cpu", "GRUNet", "tanh 3 32 32 1", 0.001);

  ////////////////////////// Acquisitor
  Acquisitor_ acquisitor;

  ////////////////////////// Algorithm ////////////////////////////////
  rai::Algorithm::PPO<Dtype, StateDim, ActionDim>
      algorithm(taskVector, &Vfunction, &policy, noiseVector, &acquisitor, 0.97, 0, 0, 1, 10, 0, false);

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

  rai::Utils::Graph::FigProp2D figurePropertiescoef;
  figurePropertiescoef.title = "Number of Steps Taken vs Kl_coeff";
  figurePropertiescoef.xlabel = "N. Steps Taken";
  figurePropertiescoef.ylabel = "Kl_coeff";

  rai::Utils::Graph::FigProp3D figurePropertiesSVC;
  figurePropertiesSVC.title = "V function";
  figurePropertiesSVC.xlabel = "angle";
  figurePropertiesSVC.ylabel = "T";
  figurePropertiesSVC.zlabel = "value";
  figurePropertiesSVC.displayType = rai::Utils::Graph::DisplayType3D::heatMap3D;

  rai::Utils::Graph::FigProp3D figurePropertiesSVA;
  figurePropertiesSVA.title = "Policy";
  figurePropertiesSVA.xlabel = "angle";
  figurePropertiesSVA.ylabel = "angular velocity";
  figurePropertiesSVA.zlabel = "action";
  figurePropertiesSVA.displayType = rai::Utils::Graph::DisplayType3D::heatMap3D;

  rai::Utils::Graph::FigPropPieChart propChart;

  ////////////////////////// Choose the computation mode //////////////

  ////////////////////////// Learning /////////////////////////////////
  constexpr int loggingInterval = 10;
  for (int iterationNumber = 0; iterationNumber < 250; iterationNumber++) {

    if (iterationNumber % loggingInterval == 0) {
      algorithm.setVisualizationLevel(1);
      taskVector[0]->enableVideoRecording();
    }
    LOG(INFO) << iterationNumber << "th Iteration";
    algorithm.runOneLoop(5000);

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


      graph->figure(2, figurePropertieskl);
      graph->appendData(2, logger->getData("klD", 0),
                        logger->getData("klD", 1),
                        logger->getDataSize("klD"),
                        rai::Utils::Graph::PlotMethods2D::linespoints,
                        "klD",
                        "lw 2 lc 4 pi 1 pt 5 ps 1");
      graph->drawFigure(2);
    }
  }

  policy.dumpParam(RAI_LOG_PATH + "/policy.txt");
  graph->drawPieChartWith_RAI_Timer(0, timer->getTimedItems(), propChart);
  graph->drawFigure(0, rai::Utils::Graph::OutputFormat::pdf);
  graph->waitForEnter();

}