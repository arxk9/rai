


#include <rai/RAI_core>

// Eigen
#include <Eigen/Dense>

// task
#include "rai/tasks/poleBalancing/PoleBalancing.hpp"

// noise model
#include "rai/noiseModel/NormalDistributionNoise.hpp"

// Neural network
#include "rai/function/tensorflow/ValueFunction_TensorFlow.hpp"
#include "rai/function/tensorflow/StochasticPolicy_TensorFlow.hpp"

// algorithm
#include "rai/algorithm/TRPO_gae.hpp"
#include <rai/experienceAcquisitor/TrajectoryAcquisitor_MultiThreadBatch.hpp>
#include <rai/experienceAcquisitor/TrajectoryAcquisitor_SingleThreadBatch.hpp>

using namespace std;
using namespace boost;

/// learning states
using Dtype = float;

/// shortcuts
using RAI::Task::ActionDim;
using RAI::Task::StateDim;
using RAI::Task::CommandDim;
using Task = RAI::Task::PoleBalancing<Dtype>;
using State = Task::State;
using StateBatch = Task::StateBatch;
using Action = Task::Action;
using ActionBatch = Task::ActionBatch;
using CostBatch = Task::CostBatch;
using VectorXD = Task::VectorXD;
using MatrixXD = Task::MatrixXD;
using Policy_TensorFlow = RAI::FuncApprox::StochasticPolicy_TensorFlow<Dtype, StateDim, ActionDim>;
using Vfunction_TensorFlow = RAI::FuncApprox::ValueFunction_TensorFlow<Dtype, StateDim>;
using Acquisitor_ = RAI::ExpAcq::TrajectoryAcquisitor_SingleThreadBatch<Dtype, StateDim, ActionDim>;
//using Acquisitor_ = RAI::ExpAcq::TrajectoryAcquisitor_MultiThreadBatch<Dtype, StateDim, ActionDim>;
using Noise = RAI::Noise::NormalDistributionNoise<Dtype, ActionDim>;
using NoiseCovariance = Eigen::Matrix<Dtype, ActionDim, ActionDim>;

#define nThread 4

int main(int argc, char *argv[]) {

  RAI_init();
  omp_set_num_threads(nThread);

  ////////////////////////// Define task ////////////////////////////
  std::vector<Task> taskVec(nThread,Task(Task::fixed, Task::easy) );
  std::vector<RAI::Task::Task<Dtype,StateDim,ActionDim,0> *> taskVector;

  for (auto& task : taskVec) {
    task.setControlUpdate_dt(0.05);
    task.setDiscountFactor(0.995);
    task.setRealTimeFactor(1.5);
    task.setTimeLimitPerEpisode(25.0);
    taskVector.push_back(&task);
  }

  ////////////////////////// Define Noise Model //////////////////////
  Dtype Stdev = 1;

  std::vector<Noise> noiseVec(nThread, Noise(NoiseCovariance::Identity()*Stdev));
  std::vector<Noise *> noiseVector;
  for(auto& noise : noiseVec)
    noiseVector.push_back(&noise);

  ////////////////////////// Define Function approximations //////////
  Vfunction_TensorFlow Vfunction("gpu,0", "MLP", "relu 1e-3 3 32 32 1", 0.001);
  Policy_TensorFlow policy("gpu,0", "MLP", "relu 1e-3 3 32 32 1", 0.001);

  ////////////////////////// Acquisitor
  Acquisitor_ acquisitor;

  ////////////////////////// Algorithm ////////////////////////////////
  RAI::Algorithm::TRPO_gae<Dtype, StateDim, ActionDim>
      algorithm(taskVector, &Vfunction, &policy, noiseVector, &acquisitor, 0.97, 0, 0, 1);
  algorithm.setVisualizationLevel(0);

  /////////////////////// Plotting properties ////////////////////////
  RAI::Utils::Graph::FigProp2D figurePropertiesEVP;
  figurePropertiesEVP.title = "Number of Steps Taken vs Performance";
  figurePropertiesEVP.xlabel = "N. Steps Taken";
  figurePropertiesEVP.ylabel = "Performance";

  RAI::Utils::Graph::FigProp2D figurePropertiesSur;
  figurePropertiesSur.title = "Number of Steps Taken vs Surrogate loss";
  figurePropertiesSur.xlabel = "N. Steps Taken";
  figurePropertiesSur.ylabel = "loss";

  RAI::Utils::Graph::FigProp3D figurePropertiesSVC;
  figurePropertiesSVC.title = "V function";
  figurePropertiesSVC.xlabel = "angle";
  figurePropertiesSVC.ylabel = "angular velocity";
  figurePropertiesSVC.zlabel = "value";
  figurePropertiesSVC.displayType = RAI::Utils::Graph::DisplayType3D::heatMap3D;

  RAI::Utils::Graph::FigProp3D figurePropertiesSVA;
  figurePropertiesSVA.title = "Policy";
  figurePropertiesSVA.xlabel = "angle";
  figurePropertiesSVA.ylabel = "angular velocity";
  figurePropertiesSVA.zlabel = "action";
  figurePropertiesSVA.displayType = RAI::Utils::Graph::DisplayType3D::heatMap3D;

  RAI::Utils::Graph::FigPropPieChart propChart;

  ////////////////////////// Choose the computation mode //////////////
  StateBatch state_plot(3, 2601);
  ActionBatch action_plot(1, 2601);
  CostBatch value_plot(1, 2601);
  MatrixXD minimal_X_extended(1, 2601);
  MatrixXD minimal_Y_extended(1, 2601);

  MatrixXD minimal_X_sampled(1, 2601);
  MatrixXD minimal_Y_sampled(1, 2601);
  ActionBatch action_sampled(1, 2601);
  MatrixXD arrowTip(1, 2601);
  MatrixXD zeros2601(1, 5601);
  zeros2601.setZero();

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

  for (int iterationNumber = 0; iterationNumber < 60; iterationNumber++) {

    if ( iterationNumber % loggingInterval == 0 ) {
      algorithm.setVisualizationLevel(1);
      taskVector[0]->enableVideoRecording();
    }

    LOG(INFO) << iterationNumber << "th Iteration";
    algorithm.runOneLoop(2000);

    if ( iterationNumber % loggingInterval == 0 ) {
      algorithm.setVisualizationLevel(0);
      graph->figure(1, figurePropertiesEVP);
      graph->appendData(1, logger->getData("Nominal performance", 0),
                        logger->getData("Nominal performance", 1),
                        logger->getDataSize("Nominal performance"),
                        RAI::Utils::Graph::PlotMethods2D::linespoints,
                        "performance",
                        "lw 2 lc 4 pi 1 pt 5 ps 1");
      graph->drawFigure(1);

      policy.forward(state_plot, action_plot);
      Vfunction.forward(state_plot, value_plot);
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
  graph->drawFigure(5, RAI::Utils::Graph::OutputFormat::pdf);
  graph->waitForEnter();

}