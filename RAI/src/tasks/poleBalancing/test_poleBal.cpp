//
// Created by joonho on 21.06.17.
//


#include <rai/RAI_core>
#include <rai/function/tensorflow/StochasticPolicy_TensorFlow.hpp>
#include "rai/tasks/poleBalancing/PoleBalancing_temp.hpp"
#include "rai/noiseModel/NormalDistributionNoise.hpp"

typedef RAI::Task::PoleBalancing<double>::State State;
typedef RAI::Task::PoleBalancing<double>::Action Action;
using Task = RAI::Task::PoleBalancing<double>;

int main(){
  using Dtype = double;
  using RAI::Task::ActionDim;
  using RAI::Task::StateDim;
  using RAI::Task::CommandDim;
  using NoiseCov = RAI::Noise::NormalDistributionNoise<Dtype, ActionDim>::CovarianceMatrix;

  RAI_init();
//  RAI::Task::QuadrupedLocomotion<double> quad(RAI_ROOT_PATH +"/RAI/taskModules/QuadrupedLocomotion/cadModel/roughterrain.obj");
  RAI::Task::PoleBalancing<double> task(Task::fixed,Task::easy);
  RAI::Noise::NormalDistributionNoise<Dtype, ActionDim> noise(NoiseCov::Identity() * Dtype(1));

  Task::Action dummyAction;
  Task::State dummyState;
  RAI::TerminationType dummyType;

  dummyAction.setZero();
  double dummyCost;

  task.turnOnVisualization("");

  task.setRealTimeFactor(1);
  task.init();

  for (int i=0; i<1000; i++) {
    State state;
    task.getState(state);
//    policy.forward(state, action);
    dummyAction.setZero();
    noise.initializeNoise();

    dummyAction += noise.sampleNoise();
//  dummyAction.setOnes();
    task.step(-dummyAction, dummyState, dummyType, dummyCost);
    if (dummyType == RAI::TerminationType::terminalState){
      dummyType = RAI::TerminationType::not_terminated;
      task.setToInitialState();
    }
  }


  RAI::Utils::Graph::FigPropPieChart propChart;
  graph->drawPieChartWith_RAI_Timer(3, timer->getTimedItems(), propChart);
  graph->drawFigure(3, RAI::Utils::Graph::OutputFormat::pdf);
  graph->waitForEnter();

}
