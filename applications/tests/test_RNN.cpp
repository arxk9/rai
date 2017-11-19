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

#include "rai/function/tensorflow/ValueFunction_TensorFlow.hpp"
#include "rai/function/common/Policy.hpp"
#include <math.h>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <functional>
#include <rai/function/tensorflow/RecurrentStochasticPolicy_TensorFlow.hpp>
#include "rai/noiseModel/NormalDistributionNoise.hpp"

#include "rai/RAI_core"
#include "rai/RAI_Tensor.hpp"

#include "rai/experienceAcquisitor/TrajectoryAcquisitor_MultiThreadBatch.hpp"
#include "rai/tasks/poleBalancing/PoleBalancing.hpp"
#include <rai/algorithm/common/LearningData.hpp>
#include <rai/algorithm/common/dataStruct.hpp>

#include <vector>
#include "rai/memory/ReplayMemoryHistory.hpp"

using namespace rai;


using std::cout;
using std::endl;
using std::cin;
using rai::Task::ActionDim;
using rai::Task::StateDim;

using Dtype = float;

using PolicyBase = rai::FuncApprox::Policy<Dtype, StateDim, ActionDim>;
using RnnQfunc = rai::FuncApprox::RecurrentQfunction_TensorFlow<Dtype, StateDim, ActionDim>;

using RnnPolicy = rai::FuncApprox::RecurrentStochasticPolicy_TensorFlow<Dtype, StateDim, ActionDim>;
using Acquisitor_ = rai::ExpAcq::TrajectoryAcquisitor_MultiThreadBatch<Dtype, StateDim, ActionDim>;
using Task_ = rai::Task::PoleBalancing<Dtype>;

using MatrixXD = Eigen::Matrix<Dtype, -1, -1>;
using VectorXD = Eigen::Matrix<Dtype, -1, 1>;
using Advantages = Eigen::Matrix<Dtype, 1, Eigen::Dynamic>;
typedef typename PolicyBase::State State;
typedef typename PolicyBase::StateBatch StateBatch;
typedef typename PolicyBase::Action Action;
typedef typename PolicyBase::ActionBatch ActionBatch;
typedef typename PolicyBase::JacobianWRTparam JacobianWRTparam;
using NormNoise = rai::Noise::NormalDistributionNoise<Dtype, ActionDim>;
using NoiseCov = Eigen::Matrix<Dtype, ActionDim, ActionDim>;
using Noise = rai::Noise::Noise<Dtype, ActionDim>;
using TensorBatchBase = rai::Algorithm::TensorBatch<Dtype>;
using TensorBatch = rai::Algorithm::history<Dtype, StateDim, ActionDim>;
using Tensor3D = Tensor<Dtype, 3>;
using Tensor2D = Tensor<Dtype, 2>;
using Tensor1D = Tensor<Dtype, 1>;
typedef Eigen::Matrix<Dtype, ActionDim, Eigen::Dynamic> JacobianQwrtActionBatch;

int main() {
  RAI_init();
  bool teststdev = false;
  bool testgradient = false;
  const int sampleN = 5;
  int Batsize = 100;
  int len = 100;
  Acquisitor_ acquisitor;
  rai::Algorithm::LearningData<Dtype, StateDim, ActionDim> ld_(&acquisitor);
//  Task_ task;
//  task.setTimeLimitPerEpisode(0.2);
//
//  NoiseCov covariance = NoiseCov::Identity();
//  NormNoise noise(covariance);

//  RnnPolicy policy("cpu", "GRUMLP", "tanh 3 5 / 8 1", 0.001);
//
//  rai::Tensor<Dtype, 3> states;
//  states.resize(StateDim, len, Batsize);
//  states.setZero();
//
//  std::vector<rai::Task::Task<Dtype, StateDim, ActionDim, 0> *> taskVector = {&task};
//  std::vector<rai::Noise::Noise<Dtype, ActionDim> *> noiseVector = {&noise};
//  ////
//  ld_.acquireTrajForNTimeSteps(taskVector, noiseVector, &policy, 50);
//
//  LOG(INFO) << ld_.Data.states.cols() << ", " << ld_.Data.states.batches();
//  std::cout << ld_.Data.states << std::endl << std::endl;
//  rai::Memory::ReplayMemoryHistory<Dtype, StateDim, ActionDim> memory(20);
//
//  for (int i = 0; i < 3; i++)
//    memory.SaveHistory(ld_.Data.states, ld_.Data.actions, ld_.Data.costs, ld_.Data.lengths, ld_.Data.termtypes);

//  TensorBatch test_bat(3, 4);
//  TensorBatch test_minibat;
//  TensorBatchBase *test_base;

//  /// Test RQFUNC
//  {
  cout << "Test: Policy::copyStructureFrom" << endl;
//
//
  RnnQfunc qfunction1("cpu", "GRUMLP2", "tanh 3 1 5 / 4 1", 0.001);

  int nIterations = 100;
  int maxlen = 10;
  int batchSize = 10;
  RandomNumberGenerator<Dtype> rn_;

  Tensor3D stateBatch;
  Tensor3D actionBatch;
  Tensor3D valueBatch;
  Tensor1D length;
  length.resize(batchSize);
  stateBatch.resize(StateDim, maxlen, batchSize);
  actionBatch.resize(ActionDim, maxlen, batchSize);
  valueBatch.resize(1, maxlen, batchSize);

  stateBatch.setRandom();
  actionBatch.setRandom();
  valueBatch.setRandom();

  rai::Algorithm::history<Dtype, StateDim, ActionDim> DATA;
  DATA.resize(maxlen, batchSize);
  DATA.setZero();
  DATA.states.copyDataFrom(stateBatch);
  DATA.actions.copyDataFrom(actionBatch);
  DATA.minibatch = new rai::Algorithm::history<Dtype, StateDim, ActionDim>;
  for (int k = 0; k < batchSize; k++) {
//    DATA.lengths[k] = 2 * (k + 1);
    DATA.lengths[k] = maxlen; //full length
  }

  LOG(INFO) << DATA.lengths.eMat();
  LOG(INFO) << DATA.states.rows() << " " << DATA.states.cols() << " " << DATA.states.batches();
  LOG(INFO) << DATA.actions.rows() << " " << DATA.actions.cols() << " " << DATA.actions.batches();
//  qfunction1.forward(DATA.states ,DATA.actions, valueBatch);


  DATA.iterateBatch(batchSize);

  for (int iteration = 0; iteration < nIterations; ++iteration) {
    Dtype loss = qfunction1.performOneSolverIter(DATA.minibatch, valueBatch);
    if (iteration % 100 == 0) cout << iteration << ", loss = " << loss << endl;
  }
//
//  RnnQfunc qfunction2("cpu", "GRUMLP2", "tanh 3 1 5 / 4 1", 0.001);
//  qfunction2.copyStructureFrom(&qfunction1);
//
//  VectorXD param1, param2;
//  qfunction1.getAP(param1);
//  qfunction2.getAP(param2);
//
//  cout << "testing interpolation" << endl;
//  cout << "from cpp calculation " << endl << (param1 * 0.2 + param2 * 0.8).transpose() << endl;
//  qfunction2.interpolateAPWith(&qfunction1, 0.2);
//  qfunction2.getAP(param2);
//  cout << "from tensorflow " << endl << param2.transpose() << endl;
//  cout << "Press Enter if the two vectors are the same" << endl;
//  cin.get();


  Tensor3D valueBatch2;

  valueBatch2.resize(1, maxlen, batchSize);

//  valueBatch2.setRandom();
  Tensor3D Gradtest;
  Tensor3D GradtestNum;
  Gradtest.resize(ActionDim, maxlen, batchSize);
  GradtestNum.resize(ActionDim, maxlen, batchSize);
  Gradtest.setZero();
  GradtestNum.setZero();

  qfunction1.test(DATA.minibatch, valueBatch);

  LOG(INFO) << qfunction1.getGradient_AvgOf_Q_wrt_action(DATA.minibatch, Gradtest);
  qfunction1.forward(DATA.minibatch->states, DATA.minibatch->actions, valueBatch);

  Dtype perturb = 1e-6;

    for (int i = 0; i < ActionDim; i++) {
      for (int k = 0; k < batchSize; k++) {
        for (int j = 0; j < DATA.minibatch->lengths[k]; j++) {

          DATA.minibatch->actions.eTensor()(i, j, k) += perturb;
          qfunction1.forward(DATA.minibatch->states, DATA.minibatch->actions, valueBatch2);
          DATA.minibatch->actions.eTensor()(i, j, k) -= perturb;
          GradtestNum.eTensor()(i, j, k) = (valueBatch2.eTensor()(0, j, k) - valueBatch.eTensor()(0, j, k)) / (perturb);
          GradtestNum.eTensor()(i, j, k) /= batchSize;
//          std::cout << j << ", " << valueBatch2.batch(0)<< std::endl;
//          std::cout << j << ", " << valueBatch3.batch(0)<< std::endl;
//          std::cout << j << ", " << valueBatch2.eTensor()(0, j, k) - valueBatch3.eTensor()(0, j, k)<< std::endl;
        }
      }
    }
//    Eigen::Matrix<Dtype,-1,-1> sss1 = Gradtest.batch(0);
//    Eigen::Matrix<Dtype,-1,-1> sss2 = GradtestNum.batch(0);

    cout << "jaco from TF is       " << endl << Gradtest << endl;
    cout << "jaco from numerical is" << endl << GradtestNum << endl;
//    cout << (sss1 - sss2).norm()/sss1.norm()*100 << ", "<< perturb << endl;

//  }
};