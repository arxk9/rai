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

#include "rai/function/tensorflow/ValueFunction_TensorFlow.hpp"
#include "rai/function/common/Policy.hpp"
#include <math.h>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <functional>
//#include <rai/function/tensorflow/RecurrentDeterministicPolicy_Tensorflow.hpp>

#include <rai/function/tensorflow/RecurrentStochasticPolicy_TensorFlow.hpp>
#include "rai/noiseModel/NormalDistributionNoise.hpp"

#include "rai/RAI_core"
#include "rai/RAI_Tensor.hpp"

#include "rai/experienceAcquisitor/TrajectoryAcquisitor_Parallel.hpp"
#include "rai/tasks/poleBalancing/PoleBalancing.hpp"
#include <rai/algorithm/common/LearningData.hpp>

#include <vector>
#include <rai/experienceAcquisitor/TrajectoryAcquisitor_Parallel.hpp>
//#include "rai/memory/ReplayMemoryHistory.hpp"

using namespace rai;


using std::cout;
using std::endl;
using std::cin;
using rai::Task::ActionDim;
using rai::Task::StateDim;
using rai::Task::CommandDim;

using Dtype = float;

using PolicyBase = rai::FuncApprox::Policy<Dtype, StateDim, ActionDim>;
using RnnQfunc = rai::FuncApprox::RecurrentQfunction_TensorFlow<Dtype, StateDim, ActionDim>;
//using RnnDPolicy = rai::FuncApprox::RecurrentDeterministicPolicy_TensorFlow<Dtype, StateDim, ActionDim>;
using RnnPolicy = rai::FuncApprox::RecurrentStochasticPolicy_TensorFlow<Dtype, StateDim, ActionDim>;
using Acquisitor_ = rai::ExpAcq::TrajectoryAcquisitor_Parallel<Dtype, StateDim, ActionDim>;
using Task_ = rai::Task::PoleBalancing<Dtype>;
using StochPol = rai::FuncApprox::StochasticPolicy_TensorFlow<Dtype, StateDim, ActionDim>;
using Value = rai::FuncApprox::ValueFunction_TensorFlow<Dtype, StateDim>;

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

using Tensor3D = Tensor<Dtype, 3>;
using Tensor2D = Tensor<Dtype, 2>;
using Tensor1D = Tensor<Dtype, 1>;
typedef Eigen::Matrix<Dtype, ActionDim, Eigen::Dynamic> JacobianQwrtActionBatch;
typedef rai::Algorithm::LearningData<Dtype, StateDim, ActionDim> TensorBatch_;
using Policy_ = FuncApprox::Policy<Dtype, StateDim, ActionDim>;

#define nThread 10

int main() {
  RAI_init();
  const int sampleN = 5;
  Acquisitor_ acquisitor;
  rai::Algorithm::LearningData<Dtype, StateDim, ActionDim> ld_;
  {
    std::vector<Task_> taskVec(nThread, Task_(Task_::fixed, Task_::easy));
    std::vector<rai::Task::Task<Dtype, StateDim, ActionDim, 0> *> taskVector;

    for (auto &task : taskVec) {
      task.setControlUpdate_dt(0.05);
      task.setDiscountFactor(0.995);
      task.setRealTimeFactor(3);
      task.setTimeLimitPerEpisode(25.0);
      taskVector.push_back(&task);
    }

    NoiseCov covariance = NoiseCov::Identity();
    std::vector<NormNoise> noiseVec(nThread, NormNoise(covariance));
    std::vector<rai::Noise::Noise<Dtype, ActionDim> *> noiseVector;
    for (auto &noise : noiseVec)
      noiseVector.push_back(&noise);

    StochPol policy("cpu", "MLP", "tanh 1e-3 3 32 32 1", 0.001);
    Value vfunction("cpu", "MLP", "tanh 1e-3 3 32 32 1", 0.001);

//    RnnPolicy policy("cpu", "GRUMLP", "tanh 1e-3 3 5 / 8 1", 0.001);
//    FuncApprox::ValueFunction_TensorFlow<Dtype, StateDim> Vfunc("cpu", "MLP", "tanh 1e-3 3 32 1", 0.001);


    Eigen::Matrix<Dtype, StateDim, -1> stateBat, stateTest, termStateBat;
    Eigen::Matrix<Dtype, 1, -1> termValueBat, valueBat, value_old;
    Tensor3D stateTensor("state");
    Tensor3D actionTensor("sampledAction");
    Tensor3D actionNoiseTensor("actionNoise");
    Tensor1D trajLength("length");

    ld_.miniBatch = new rai::Algorithm::LearningData<Dtype, StateDim, ActionDim>;
    acquisitor.setData(&ld_);
    Tensor<Dtype, 2> valuePred("predictedValue");
    ld_.append(valuePred);

    for (int i = 0; i < 10; i++) {
      std::cout << "iter" << i << std::endl;
      acquisitor.acquireVineTrajForNTimeSteps(taskVector, noiseVector, &policy, 6000, 0, 0, &vfunction);
      acquisitor.saveData(taskVector[0], &policy, &vfunction);

      Dtype disc = taskVector[0]->discountFtr();
      int dataN = 0;
      for (auto &tra : acquisitor.traj) dataN += tra.size() - 1;
      stateBat.resize(StateDim, dataN);
      stateTest.resize(StateDim, dataN);
      int colID = 0;
      for (int traID = 0; traID < acquisitor.traj.size(); traID++) {
        for (int timeID = 0; timeID < acquisitor.traj[traID].size() - 1; timeID++) {
          stateBat.col(colID++) = acquisitor.traj[traID].stateTraj[timeID];
        }
      }

      stateTensor.resize(StateDim, 1, dataN);
      actionTensor.resize(ActionDim, 1, dataN);
      actionNoiseTensor.resize(ActionDim, 1, dataN);
      stateTensor.setZero();
      actionTensor.setZero();
      actionNoiseTensor.setZero();
      trajLength.resize(1);
      trajLength[0] = dataN;

      int pos = 0;
      for (int traID = 0; traID < acquisitor.traj.size(); traID++) {
        for (int timeID = 0; timeID < acquisitor.traj[traID].size() - 1; timeID++) {
          actionTensor.batch(pos) = acquisitor.traj[traID].actionTraj[timeID];
          actionNoiseTensor.batch(pos++) = acquisitor.traj[traID].actionNoiseTraj[timeID];
        }
      }
      stateTensor.copyDataFrom(stateBat);

      termValueBat.resize(1, acquisitor.traj.size());
      termStateBat.resize(StateDim, acquisitor.traj.size());
      valueBat.resize(dataN);

      for (int traID = 0; traID < acquisitor.traj.size(); traID++)
        termStateBat.col(traID) = acquisitor.traj[traID].stateTraj.back();

      vfunction.forward(termStateBat, termValueBat);

      for (int traID = 0; traID < acquisitor.traj.size(); traID++) {
        if (acquisitor.traj[traID].termType == TerminationType::timeout) {
          acquisitor.traj[traID].updateValueTrajWithNewTermValue(termValueBat(traID), disc);
        }
      }
      colID = 0;
      for (int traID = 0; traID < acquisitor.traj.size(); traID++) {
        for (int timeID = 0; timeID < acquisitor.traj[traID].size() - 1; timeID++)
          valueBat(colID++) = acquisitor.traj[traID].valueTraj[timeID];
      }

      std::cout << "states err" << (ld_.states.eTensor() - stateTensor.eTensor()).sum() << std::endl;
      std::cout << "actions err" << (ld_.actions.eTensor() - actionTensor.eTensor()).sum() << std::endl;
      std::cout << "actionNoise err" << (ld_.actionNoises.eTensor() - actionNoiseTensor.eTensor()).sum()
                << std::endl;
      ld_.extraTensor2D[0].resize(ld_.maxLen, ld_.batchNum);

      vfunction.forward(ld_.states, ld_.extraTensor2D[0]);
      value_old.resize(stateBat.cols());
      vfunction.forward(stateBat, value_old);

      Action stdev_o;
      policy.getStdev(stdev_o);
      Eigen::Matrix<Dtype, -1, 1> policy_grad = Eigen::Matrix<Dtype, -1, 1>::Zero(policy.getLPSize());
      Eigen::Matrix<Dtype, -1, 1> policy_grad2 = Eigen::Matrix<Dtype, -1, 1>::Zero(policy.getLPSize());
      ///advantage
      acquisitor.computeAdvantage(taskVector[0], &vfunction, 0.97, true);
      Tensor2D advantages2("advantage");
      advantages2.resize(1, dataN);

      int dataID = 0;
      for (int trajID = 0; trajID < acquisitor.traj.size(); trajID++) {

        Eigen::Matrix<Dtype, 1, -1> valueMat;
        Eigen::Matrix<Dtype, 1, -1> bellmanErr;
        Eigen::Matrix<Dtype, 1, -1> advs;
        Eigen::Matrix<Dtype, StateDim, -1> stateTrajMat;
        Eigen::Matrix<Dtype, ActionDim, -1> actionTrajMat;

        valueMat.resize(acquisitor.traj[trajID].size());
        bellmanErr.resize(acquisitor.traj[trajID].size() - 1);
        stateTrajMat.resize(StateDim, acquisitor.traj[trajID].size());
        actionTrajMat.resize(ActionDim, acquisitor.traj[trajID].size());
        for (int col = 0; col < acquisitor.traj[trajID].size(); col++) {
          stateTrajMat.col(col) = acquisitor.traj[trajID].stateTraj[col];
          actionTrajMat.col(col) = acquisitor.traj[trajID].actionTraj[col];
        }
        vfunction.forward(stateTrajMat, valueMat);

        if (acquisitor.traj[trajID].termType == TerminationType::terminalState)
          valueMat[acquisitor.traj[trajID].size() - 1] = taskVector[0]->termValue();

        for (int w = 0; w < acquisitor.traj[trajID].size() - 1; w++)
          bellmanErr[w] =
              valueMat[w + 1] * taskVector[0]->discountFtr() + acquisitor.traj[trajID].costTraj[w] - valueMat[w];

        advs.resize(1, acquisitor.traj[trajID].size() - 1);
        advs[acquisitor.traj[trajID].size() - 2] = bellmanErr[acquisitor.traj[trajID].size() - 2];
        Dtype fctr = taskVector[0]->discountFtr() * 0.97;

        for (int timeID = acquisitor.traj[trajID].size() - 3; timeID > -1; timeID--)
          advs[timeID] = fctr * advs[timeID + 1] + bellmanErr[timeID];

        rai::Math::MathFunc::normalize(advs);

        advantages2.block(0, dataID, 1, advs.cols()) = advs;
        dataID += advs.cols();
      }


      ///test epoch
      for (int k = 0; k < 10; k++) {
        while (ld_.iterateBatch(0)) {
          std::cout << "value" << (ld_.miniBatch->values.eMat() - valueBat).sum();
          std::cout << " value2" << (ld_.miniBatch->extraTensor2D[0].eMat() - value_old).sum();
          std::cout << " states" << (ld_.miniBatch->states.eTensor() - stateTensor.eTensor()).sum();
          std::cout << " actions" << (ld_.miniBatch->actions.eTensor() - actionTensor.eTensor()).sum();
          std::cout << " actionNoise" << (ld_.miniBatch->actionNoises.eTensor() - actionNoiseTensor.eTensor()).sum()
                    << std::endl;
          std::cout << " advs" << (ld_.miniBatch->advantages.eMat() - advantages2.eMat()).norm()/advantages2.eMat().norm() << std::endl;

          policy.PPOpg(ld_.miniBatch, stdev_o, policy_grad);
          policy.PPOpg(stateTensor, actionTensor, actionNoiseTensor, advantages2, stdev_o, ld_.lengths, policy_grad2);
          std::cout << " grad" << (policy_grad - policy_grad2).norm()/policy_grad2.norm()  << std::endl;

        }
      }
    }
  }

};