//
// Created by joonho on 11/12/17.
//


#include <rai/RAI_core>

#include <iostream>
#include "rai/function/tensorflow/common/ParameterizedFunction_TensorFlow.hpp"
#include "rai/function/tensorflow/Qfunction_TensorFlow.hpp"
#include "rai/function/tensorflow/DeterministicPolicy_TensorFlow.hpp"
#include "rai/function/tensorflow/StochasticPolicy_TensorFlow.hpp"
#include "rai/function/tensorflow/RecurrentQfunction_TensorFlow.hpp"
#include <rai/noiseModel/NoNoise.hpp>
#include "rai/function/tensorflow/RecurrentValueFunction_TensorFlow.hpp"

#include "rai/function/tensorflow/ValueFunction_TensorFlow.hpp"
#include "rai/function/common/Policy.hpp"
#include <math.h>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
//#include <rai/function/tensorflow/RecurrentDeterministicPolicy_Tensorflow.hpp>
#include <rai/function/tensorflow/RecurrentStochasticPolicy_TensorFlow.hpp>
#include "rai/noiseModel/NormalDistributionNoise.hpp"

#include "rai/RAI_core"
#include "rai/RAI_Tensor.hpp"

#include "rai/experienceAcquisitor/TrajectoryAcquisitor_Parallel.hpp"
//#include "rai/tasks/poleBalancing/PoleBalancing.hpp"
#include "rai/tasks/poleBalancing/POPoleBalancing.hpp"

#include <rai/algorithm/common/LearningData.hpp>

using namespace rai;


using std::cout;
using std::endl;
using std::cin;
constexpr int StateDim = 2;
constexpr int ActionDim = 1;

using Dtype = double;

using PolicyBase = rai::FuncApprox::Policy<Dtype, StateDim, ActionDim>;
using RnnVfunc = rai::FuncApprox::RecurrentValueFunction_TensorFlow<Dtype, StateDim>;
//using RnnQfunc = rai::FuncApprox::RecurrentQfunction_TensorFlow<Dtype, StateDim, ActionDim>;
//using RnnDPolicy = rai::FuncApprox::RecurrentDeterministicPolicy_TensorFlow<Dtype, StateDim, ActionDim>;
using RnnPolicy = rai::FuncApprox::RecurrentStochasticPolicy_TensorFlow<Dtype, StateDim, ActionDim>;
using Acquisitor_ = rai::ExpAcq::TrajectoryAcquisitor_Parallel<Dtype, StateDim, ActionDim>;
//using Task_ = rai::Task::PoleBalancing<Dtype>;
using Task_ = rai::Task::PO_PoleBalancing<Dtype>;

using MatrixXD = Eigen::Matrix<Dtype, -1, -1>;
using VectorXD = Eigen::Matrix<Dtype, -1, 1>;
using Advantages = Eigen::Matrix<Dtype, 1, Eigen::Dynamic>;
typedef typename PolicyBase::State State;
typedef typename PolicyBase::StateBatch StateBatch;
typedef typename PolicyBase::Action Action;
typedef typename PolicyBase::ActionBatch ActionBatch;
typedef typename PolicyBase::JacobianWRTparam JacobianWRTparam;
using NormNoise = rai::Noise::NormalDistributionNoise<Dtype, ActionDim>;
using NoNoise = rai::Noise::NoNoise<Dtype, ActionDim>;

using NoiseCov = Eigen::Matrix<Dtype, ActionDim, ActionDim>;
using Noise = rai::Noise::Noise<Dtype, ActionDim>;
//using TensorBatchBase = rai::Algorithm::TensorBatch<Dtype>;
//using TensorBatch = rai::Algorithm::historyWithAdvantage<Dtype, StateDim, ActionDim>;
using Tensor3D = Tensor<Dtype, 3>;
using Tensor2D = Tensor<Dtype, 2>;
using Tensor1D = Tensor<Dtype, 1>;
typedef Eigen::Matrix<Dtype, ActionDim, Eigen::Dynamic> JacobianQwrtActionBatch;
//typedef rai::Algorithm::historyWithAdvantage<Dtype, StateDim, ActionDim>  TensorBatch_;
using Policy_ = FuncApprox::Policy<Dtype, StateDim, ActionDim>;

int main() {
  RAI_init();
  int colID = 0;
int iterN = 30;

  Acquisitor_ acquisitor;
  rai::Algorithm::LearningData<Dtype, StateDim, ActionDim> ld_;

  Task_ task;
  task.setTimeLimitPerEpisode(5);

  NoiseCov covariance = NoiseCov::Identity();
  NormNoise noise(covariance);

  RnnPolicy policy("cpu", "GRUMLP", "tanh 1e-3 2 5 / 8 1", 0.001);
//  RnnVfunc Vfunc("gpu,0", "GRUNet", "tanh 2 64 32 1", 0.01);
//  RnnVfunc Vfunc2("gpu,0", "GRUNet", "tanh 2 64 32 1", 0.01);
  RnnVfunc Vfunc("gpu,0", "LSTMMLP", "tanh 1e-3 2 64 / 32 32 1", 0.01);
  RnnVfunc Vfunc2("gpu,0", "LSTMMLP", "tanh 1e-3 2 64 / 32 32 1", 0.01);

  Vfunc2.copyAPFrom(&Vfunc);

  std::vector<rai::Task::Task<Dtype, StateDim, ActionDim, 0> *> taskVector = {&task};
  std::vector<rai::Noise::Noise<Dtype, ActionDim> *> noiseVector = {&noise};

  acquisitor.setData(&ld_);
  acquisitor.acquireVineTrajForNTimeSteps(taskVector, noiseVector, &policy, 10000, 0, 0, &Vfunc);
  acquisitor.saveData(taskVector[0], &policy, &Vfunc);

  Tensor2D test1("predictedValue");
  test1.resize(ld_.maxLen, ld_.batchNum);
  std::cout << ld_.maxLen << ", " << ld_.batchNum << std::endl;
  std::cout << acquisitor.stepsTaken() << "steps" << std::endl;


  ///Visualize Data
  rai::Tensor<Dtype, 3> testv;
  rai::Tensor<Dtype, 2> v_plot;
  v_plot.resize(ld_.maxLen, ld_.batchNum);
  MatrixXD state_plot0, state_plot1, v_plot0, v_plot1, N, loss;

  state_plot0.resize(1, ld_.maxLen * ld_.batchNum);
  state_plot1.resize(1, ld_.maxLen * ld_.batchNum);
  v_plot0.resize(1, ld_.maxLen * ld_.batchNum);
  v_plot1.resize(1, ld_.maxLen * ld_.batchNum);
  N.resize(1, iterN);
  loss.resize(1, iterN);


  rai::Utils::Graph::FigProp3D figprop;
  figprop.title = "V function";
  figprop.xlabel = "angle";
  figprop.ylabel = ",";
  figprop.zlabel = "value";

  rai::Utils::Graph::FigProp2D figprop2;
  figprop2.title = "Learning curve";
  figprop2.xlabel = "iter";
  figprop2.ylabel = "cost";


  Dtype loss_sq, loss_tr;

////////////////test squared
  for (int iter = 0; iter < iterN; iter++) {
    Vfunc2.forward(ld_.states, test1);
  for (int epoch = 0; epoch < 10; epoch++) {
    loss_sq = Vfunc2.performOneSolverIter(ld_.states, ld_.values, ld_.lengths);
    LOG(INFO) << "loss_sq" << loss_sq;
  }
    loss(0,iter) = loss_sq;
    N(0,iter) = iter;

  }
  Vfunc2.forward(ld_.states, v_plot);
  colID = 0;
  for (int i = 0; i < ld_.batchNum; i++) {
    for (int t = 0; t < ld_.maxLen; t++) {
      v_plot0(colID) = ld_.values.eMat()(t, i);
      v_plot1(colID) = v_plot.eMat()(t, i);

      state_plot0(colID) = std::atan2(ld_.states.eTensor()(0, t, i), ld_.states.eTensor()(1, t, i));
      state_plot1(colID++) = t;
    }
  }

  Utils::graph->figure3D(1, figprop);
  Utils::graph->append3D_Data(1, state_plot0.data(), state_plot1.data(), v_plot0.data(), v_plot0.cols(),
                              false, Utils::Graph::PlotMethods3D::points, "groundtruth");

  Utils::graph->append3D_Data(1, state_plot0.data(), state_plot1.data(), v_plot1.data(), v_plot0.cols(),
                              false, Utils::Graph::PlotMethods3D::points, "learned");
  Utils::graph->drawFigure(1);

  Utils::graph->figure(3, figprop2);
  Utils::graph->appendData(3, N.data(), loss.data(),loss.cols(), "");

  Utils::graph->drawFigure(3);


////////////////test Trustregion method
  for (int iter = 0; iter < iterN; iter++) {
    for (int epoch = 0; epoch < 10; epoch++) {

    loss_tr = Vfunc.performOneSolverIter_trustregion(ld_.states,
                                                     ld_.values,
                                                     test1,
                                                     ld_.lengths);
    LOG(INFO) << "loss_tr" << loss_tr;
  }
  loss(0,iter) = loss_tr;
    N(0,iter) = iter;
  }
  Vfunc.forward(ld_.states, v_plot);
  colID = 0;
  for (int i = 0; i < ld_.batchNum; i++) {
    for (int t = 0; t < ld_.maxLen; t++) {
      v_plot1(colID++) = v_plot.eMat()(t, i);
    }
  }

  Utils::graph->figure3D(2, figprop);
  Utils::graph->append3D_Data(2, state_plot0.data(), state_plot1.data(), v_plot0.data(), v_plot0.cols(),
                              false, Utils::Graph::PlotMethods3D::points, "groundtruth");

  Utils::graph->append3D_Data(2, state_plot0.data(), state_plot1.data(), v_plot1.data(), v_plot0.cols(),
                              false, Utils::Graph::PlotMethods3D::points, "learned");
  Utils::graph->drawFigure(2);

  Utils::graph->figure(4, figprop2);
  Utils::graph->appendData(4, N.data(), loss.data(),loss.cols(), "");
  Utils::graph->drawFigure(4);
//    Utils::graph->waitForEnter();


  ///////////////////////////////////////////////////////////////////////////////////////////////

  RnnVfunc vfunction1("gpu,0", "GRUNet", "tanh 1 64 1", 0.01);
  RnnVfunc vfunction2("gpu,0", "GRUNet", "tanh 1 64 1", 0.01);

  Tensor3D state_("state");
  Tensor2D value_target("targetValue");
  Tensor2D value_predicted("predictedValue");

  Tensor1D lengths("length");

  RandomNumberGenerator<Dtype> rn_v;
  Utils::Graph::FigProp3D figure1properties("state", "T", "value", "Vfunction Squared error");
  Utils::Graph::FigProp3D figure1properties2("state", "T", "value", "Vfunction Trustregion iter");

  int len = 128;
  int batsize = 100;
  int iterN2 = 50;
  ///plot

  MatrixXD state_dim0, state_dim1, value_dat;
  MatrixXD state_dim02, state_dim12, value_dat2, value_dat3;

  state_.resize(1, len, batsize);
  value_target.resize(len, batsize);
  value_predicted.resize(len, batsize);

  lengths.resize(batsize);
  lengths.setConstant(len);
  Dtype loss1, loss2;

  for (int iter = 0; iter < iterN2; iter++) {

    for (int b = 0; b < batsize; b++) {
      for (int t = 0; t < len; t++) {
        Dtype state = 5 * rn_v.sampleNormal();
        Dtype Noise = rn_v.sampleNormal();
//        Dtype targV = std::sin(0.01 * t * 2 * M_PI) ;

        Dtype targV = std::sin(0.01 * t * 2 * M_PI) * std::sin(state * 0.25);
//        Dtype targV = std::sin(0.01 * t * 2 * M_PI) * std::sin(state * 0.25) + Noise;

        state_.eTensor()(0, t, b) = state;
        value_target.eTensor()(t, b) = targV;
      }
    }

    ///train for 10 epoch
    vfunction2.forward(state_, value_predicted);
    for (int k = 0; k < 10; k++) {
      loss1 = vfunction1.performOneSolverIter(state_, value_target, lengths);
      loss2 = vfunction2.performOneSolverIter_trustregion(state_, value_target, value_predicted, lengths);
    }
    std::cout << "Squared error iter :" << iter << " loss :" << loss1 << std::endl;
    std::cout << "Trust Region  iter :" << iter << " loss :" << loss2 << std::endl;
  }

  state_.resize(1, len, batsize);

  state_dim0.resize(1, batsize * len);
  state_dim1.resize(1, batsize * len);
  value_dat.resize(1, batsize * len);
  state_dim02.resize(1, batsize * len);
  state_dim12.resize(1, batsize * len);
  value_dat2.resize(1, batsize * len);
  value_dat3.resize(1, batsize * len);

  colID = 0;
  for (int b = 0; b < batsize; b++) {
    for (int t = 0; t < len; t++) {
      Dtype state = 5 * rn_v.sampleNormal();
      Dtype targV = std::sin(0.01 * t * 2 * M_PI) * std::sin(state * 0.25);
//      Dtype targV = std::sin(0.01 * t * 2 * M_PI) ;

      value_dat(0, colID) = targV;
      state_dim0(0, colID) = state;
      state_dim1(0, colID) = t;

      state_.eTensor()(0, t, b) = state;
      state_dim02(0, colID) = state;
      state_dim12(0, colID++) = t;
    }
  }
  Tensor2D value_predict("value");
  value_predict.resize(len, batsize);
  Tensor2D value_predict2("value");
  value_predict2.resize(len, batsize);

  vfunction1.forward(state_, value_predict);
  vfunction2.forward(state_, value_predict2);

  colID = 0;
  for (int b = 0; b < batsize; b++) {
    for (int t = 0; t < len; t++) {
      value_dat2(0, colID) = value_predict.eMat()(t, b);
      value_dat3(0, colID++) = value_predict2.eMat()(t, b);
    }
  }
  Utils::graph->figure3D(3, figure1properties);
  Utils::graph->append3D_Data(3,
                              state_dim0.data(),
                              state_dim1.data(),
                              value_dat.data(),
                              value_dat.cols(),
                              false,
                              Utils::Graph::PlotMethods3D::points,
                              "groundtruth");
  Utils::graph->append3D_Data(3,
                              state_dim02.data(),
                              state_dim12.data(),
                              value_dat2.data(),
                              value_dat2.cols(),
                              false,
                              Utils::Graph::PlotMethods3D::points,
                              "Output");
  Utils::graph->drawFigure(3);

  Utils::graph->figure3D(4, figure1properties2);
  Utils::graph->append3D_Data(4,
                              state_dim0.data(),
                              state_dim1.data(),
                              value_dat.data(),
                              value_dat.cols(),
                              false,
                              Utils::Graph::PlotMethods3D::points,
                              "groundtruth");
  Utils::graph->append3D_Data(4,
                              state_dim02.data(),
                              state_dim12.data(),
                              value_dat3.data(),
                              value_dat3.cols(),
                              false,
                              Utils::Graph::PlotMethods3D::points,
                              "Output");
  Utils::graph->drawFigure(4);

  Utils::graph->waitForEnter();



//  /// Test RQFUNC
//  {
//  RnnQfunc qfunction1("cpu", "GRUMLP2", "tanh 1e-3 3 3 5 / 10 1", 0.001);
//
//  int nIterations = 500;
//  int maxlen = 10;
//  int batchSize = 15;
//  RandomNumberGenerator<Dtype> rn_;
//
//  Tensor3D stateBatch;
//  Tensor3D actionBatch;
//  Tensor3D valueBatch;
//  Tensor1D length;
//  length.resize(batchSize);
//  stateBatch.resize(StateDim, maxlen, batchSize);
//  actionBatch.resize(ActionDim, maxlen, batchSize);
//  valueBatch.resize(1, maxlen, batchSize);
//
//  stateBatch.setRandom();
//  actionBatch.setRandom();
//  valueBatch.setRandom();
//
//  rai::Algorithm::history<Dtype, StateDim, ActionDim> DATA;
//  DATA.resize(maxlen, batchSize);
//  DATA.setZero();
//
//  DATA.minibatch = new rai::Algorithm::history<Dtype, StateDim, ActionDim>;
//
//  for (int k = 0; k < batchSize; k++) {
////    DATA.lengths[k] = 2 * (k + 1);
//    DATA.lengths[k] = maxlen; //full length
//  }
//
//  LOG(INFO) << DATA.states.rows() << " " << DATA.states.cols() << " " << DATA.states.batches();
//  LOG(INFO) << DATA.actions.rows() << " " << DATA.actions.cols() << " " << DATA.actions.batches();
////  qfunction1.forward(DATA.states ,DATA.actions, valueBatch);
//
//
//  ///Test Pol
//
//  policy.forward(DATA.states, DATA.actions);
//  policy2.forward(DATA.states, DATA.actions);
//
//  LOG(INFO) << "f";
//
//  ///Test Vfunc
//  Dtype loss = 0;
//  for (int iteration = 0; iteration < nIterations; ++iteration) {
//
//    stateBatch.setRandom();
//    actionBatch.setRandom();
//    DATA.states.copyDataFrom(stateBatch);
//    DATA.actions.copyDataFrom(actionBatch);
//    valueBatch.setZero();
//
//    for (int k = 0; k < batchSize; k++) {
//      for (int j = 0; j < DATA.lengths[k]; j++) {
//        for (int i = 0; i < 3; i++) {
//          valueBatch.eTensor()(0, j, k) += std::sin(5 * actionBatch.eTensor()(i, j, k)) - stateBatch.eTensor()(i, j, k);
//        }
//      }
//    }
//    DATA.iterateBatch(batchSize);
//    loss = qfunction1.performOneSolverIter(DATA.minibatch, valueBatch);
//    if (iteration % 100 == 0) cout << iteration << ", loss = " << loss << endl;
//  }
////  qfunction1.test(DATA.minibatch, valueBatch);
//
//  Tensor3D valueBatch2;
//
//  valueBatch2.resize(1, maxlen, batchSize);
//
////  valueBatch2.setRandom();
//  Tensor3D Gradtest;
//  Tensor3D GradtestNum;
//  Gradtest.resize(ActionDim, maxlen, batchSize);
//  GradtestNum.resize(ActionDim, maxlen, batchSize);
//  Gradtest.setZero();
//  GradtestNum.setZero();
//
//
//  qfunction1.forward(DATA.minibatch->states, DATA.minibatch->actions, valueBatch);
//
//  Dtype perturb = 5e-3;
//  int cnt=0;
//  for (int k = 0; k < batchSize; k++) {
//    cnt += DATA.minibatch->lengths[k] * ActionDim;
//  }
//  Eigen::Tensor<Dtype, 0 > temp;
//  for (int i = 0; i < ActionDim; i++) {
//    for (int k = 0; k < batchSize; k++) {
//      for (int j = 0; j < DATA.minibatch->lengths[k]; j++) {
//
//        DATA.minibatch->actions.eTensor()(i, j, k) += perturb;
//        qfunction1.forward(DATA.minibatch->states, DATA.minibatch->actions, valueBatch2);
//        DATA.minibatch->actions.eTensor()(i, j, k) -= perturb;
////        Dtype f1= valueBatch2.eTensor().sum();
//        temp = valueBatch2.eTensor().mean();
//        GradtestNum.eTensor()(i, j, k) = temp(0);
//        temp = valueBatch.eTensor().mean();
////        LOG(INFO) << temp(0) - GradtestNum.eTensor()(i, j, k) ;
//        GradtestNum.eTensor()(i, j, k) -= temp(0);
//        GradtestNum.eTensor()(i,j,k) /= perturb;
//      }
//    }
//  }
//
//  qfunction1.getGradient_AvgOf_Q_wrt_action(DATA.minibatch, Gradtest);
//
//  cout << "jaco from TF is       " << endl << Gradtest << endl;
//  cout << "jaco from numerical is" << endl << GradtestNum << endl;
//
//  RnnDPolicy DPolicy("cpu", "GRUMLP", "tanh 1e-3 3 5 / 10 3", 0.001);
//
//
//  }
};