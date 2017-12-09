//
// Created by joonho on 11/16/17.
//

//
// Created by joonho on 23.03.17.
//


#include <iostream>
#include "rai/function/tensorflow/common/ParameterizedFunction_TensorFlow.hpp"
#include "rai/function/tensorflow/Qfunction_TensorFlow.hpp"
#include "rai/function/tensorflow/DeterministicPolicy_TensorFlow.hpp"
#include "rai/function/tensorflow/StochasticPolicy_TensorFlow.hpp"
#include "rai/function/tensorflow/ValueFunction_TensorFlow.hpp"
#include "rai/function/common/Policy.hpp"
#include "raiCommon/math/RAI_math.hpp"

#include <math.h>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <functional>

#include "rai/RAI_core"
#include "rai/RAI_Tensor.hpp"
#include <rai/experienceAcquisitor/TrajectoryAcquisitor_Parallel.hpp>
#include "rai/algorithm/PPO.hpp"
#include "rai/tasks/poleBalancing/PoleBalancing.hpp"

using std::cout;
using std::endl;
using std::cin;
const int ActionDim = 1;
const int StateDim = 3;
using Dtype = float;

using NormNoise = rai::Noise::NormalDistributionNoise<Dtype, ActionDim>;
using NoiseCov = Eigen::Matrix<Dtype, ActionDim, ActionDim>;

using PolicyBase = rai::FuncApprox::Policy<Dtype, StateDim, ActionDim>;
using stPolicy = rai::FuncApprox::StochasticPolicy_TensorFlow<Dtype, StateDim, ActionDim>;

using MatrixXD = Eigen::Matrix<Dtype, -1, -1>;
using VectorXD = Eigen::Matrix<Dtype, -1, 1>;
using Advantages = Eigen::Matrix<Dtype, 1, Eigen::Dynamic>;
typedef typename PolicyBase::State State;
typedef typename PolicyBase::StateBatch StateBatch;
typedef typename PolicyBase::Action Action;
typedef typename PolicyBase::ActionBatch ActionBatch;
typedef typename PolicyBase::JacobianWRTparam JacobianWRTparam;
using Task_ = rai::Task::PoleBalancing<Dtype>;

using Vfunction_TensorFlow = rai::FuncApprox::ValueFunction_TensorFlow<Dtype, StateDim>;
using Acquisitor_ = rai::ExpAcq::TrajectoryAcquisitor_Parallel<Dtype, StateDim, ActionDim>;

using namespace rai;
//
//Dtype sample(double dummy) {
//  static std::mt19937 rng;
//  static std::normal_distribution<Dtype> nd(training_mean, sqrt(training_variance));
//  return nd(rng);
//}
#define nThread 2

int main() {
  RAI_init();
//  bool teststdev = false;
//  bool testgradient = false;
//  const int sampleN = 5;
//  stPolicy policy("gpu,0", "MLP", "relu 1e-5 3 32 32 32 1", 0.001);
//
//
//  StateBatch stateBatch = StateBatch::Random(StateDim, sampleN);
//  ActionBatch actionBatch;
//  ActionBatch actionNoise = ActionBatch::Random(ActionDim, sampleN);
//  ActionBatch actionmean;
//  ActionBatch actionmeanNew;
//  Action Stdev_o, StdevNew;
//  State state = State::Ones(StateDim);
//  Action action, action2;
//
//  VectorXD parameter, testgrad, temp;
//  MatrixXD test;
//  Advantages advs = Eigen::Matrix<Dtype, 1, sampleN>::Random(1, sampleN);
//  Dtype loss, loss2;
//
//  policy.forward(stateBatch, actionmean);
//  std::cout << actionmean <<std::endl;
//
//  actionBatch = actionmean + actionNoise;
//
//  rai::Tensor<Dtype, 3> ActionTensor;
//  rai::Tensor<Dtype, 3> ActionNTensor;
//
//  rai::Tensor<Dtype, 3> StateTensor3;
//  rai::Tensor<Dtype, 1> len;
//
////  StateTensor = "state";
//  StateTensor3 = "state";
//  ActionTensor = "sampled_oa";
//  ActionNTensor = "noise_oa";
//  len = "length";
//  StateTensor3.resize(stateBatch.rows(),1,stateBatch.cols());
//  StateTensor3.copyDataFrom(stateBatch);
//
////  StateTensor.resize(stateBatch.rows(),stateBatch.cols());
////  StateTensor = stateBatch;
//  ActionTensor.resize(actionBatch.rows(),1,actionBatch.cols());
//  ActionNTensor.resize(actionBatch.rows(),1,actionBatch.cols());
//  Stdev_o.setConstant(1);
//  ActionTensor.setConstant(1);
//
//  Utils::logger->addVariableToLog(2, "Stdev", "");
//
//  for (int i = 0; i<1e3 ; i++) {
//    timer->startTimer("dd");
//    policy.forward(StateTensor3, ActionTensor);
//    policy.getLearningRate();
//    policy.PPOpg(StateTensor3,ActionTensor,ActionNTensor,advs,Stdev_o,len,parameter);
//    policy.PPOpg_kladapt(StateTensor3,ActionTensor,ActionNTensor,advs,Stdev_o,len,parameter);
//    policy.getLP(parameter);
//    policy.setPPOparams(0.01, 0.1, 0.2);
//
//    parameter.setZero(policy.getLPSize());
//    policy.setStdev(Stdev_o);
//    policy.trainUsingGrad(parameter);
//    policy.PPOgetkl(StateTensor3, ActionTensor, ActionNTensor, Stdev_o, len);
//
//    Utils::logger->appendData("Stdev", i, Stdev_o.norm());
//    timer->stopTimer("dd");
//  }
//  LOG(INFO) << "policy check";
//
//
//  /////////////////////// Plotting properties ////////////////////////
//  rai::Utils::Graph::FigProp2D figurePropertieskl("N. Steps Taken", "d", "Number of Steps Taken vs d");
  rai::Utils::Graph::FigPropPieChart propChart;
//
//
////  policy.forward(state_plot, action_plot);
//  graph->figure(2, figurePropertieskl);
//  graph->appendData(2, logger->getData("Stdev", 0),
//                    logger->getData("Stdev", 1),
//                    logger->getDataSize("Stdev"),
//                    rai::Utils::Graph::PlotMethods2D::linespoints,
//                    "klD",
//                    "lw 2 lc 4 pi 1 pt 5 ps 1");
//  graph->drawFigure(2);
//
//
//  graph->drawPieChartWith_RAI_Timer(0, timer->getTimedItems(), propChart);
//  std::cout << ActionTensor;
//
//
//  parameter.setZero(policy.getLPSize());
//  testgrad.Random(policy.getLPSize());
//  policy.getLP(parameter);

  //////////////////////////
  Task_ testtask;
  testtask.turnOffVisualization();
//  std::vector<Task_> taskVec(nThread, Task_(Task_::fixed, Task_::easy));
//  std::vector<rai::Task::Task<Dtype, StateDim, ActionDim, 0> *> taskVector;
//
//  for (auto &task : taskVec) {
//    task.setControlUpdate_dt(0.05);
//    task.setDiscountFactor(0.995);
//    task.setRealTimeFactor(1.5);
//    task.setTimeLimitPerEpisode(25.0);
//    taskVector.push_back(&task);
//  }

//  Dtype Stdev = 1;
//  NoiseCov covariance = NoiseCov::Identity() * Stdev;
//  std::vector<NormNoise> noiseVec(nThread, NormNoise(covariance));
//  std::vector<NormNoise *> noiseVector;
//  for (auto &noise : noiseVec)
//    noiseVector.push_back(&noise);

//  Vfunction_TensorFlow Vfunction("gpu,0", "MLP", "relu 1e-3 3 32 32 1", 0.001);
//  stPolicy policy2("cpu", "MLP", "relu 1e-3 3 32 32 1", 0.001);
//  Acquisitor_ acquisitor;

//  rai::Algorithm::PPO<Dtype, StateDim, ActionDim>
//      algorithm(taskVector, &Vfunction, &policy, noiseVector, &acquisitor, 0.97, 0, 0, 1, 10, true);

//  algorithm.setVisualizationLevel(0);
//  for (int iterationNumber = 1; iterationNumber <= 10; iterationNumber++) {
//    LOG(INFO) << iterationNumber << "th Iteration";
//    algorithm.runOneLoop(100);
//  }

//  policy.dumpParam(RAI_LOG_PATH + "/policy.txt");
  graph->drawPieChartWith_RAI_Timer(0, timer->getTimedItems(), propChart);
  graph->drawFigure(0, rai::Utils::Graph::OutputFormat::pdf);
  graph->waitForEnter();

}


//
//logp_n = - 0.5 * tf.reduce_sum(tf.square((OldActionSampled - action_mean) / action_stdev), axis=1) \
//                     - 0.5 * tf.cast(tf.log(2.0 * np.pi),dtype) * action_dim - tf.reduce_sum(tf.log(action_stdev))
//logp_old = - 0.5 * tf.reduce_sum(tf.square((OldActionNoise) / OldStdv), axis=1) \
//                     - 0.5 * tf.cast(tf.log(2.0 * np.pi),dtype) * action_dim - tf.reduce_sum(tf.log(OldStdv))
//
//ratio = tf.exp(logp_n - logp_old, name='rat')
//surr = - tf.reduce_mean(tf.multiply(ratio, advant), name='Surr')