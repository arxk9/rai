//
// Created by joonho on 15.05.17.
//

#ifndef RAI_PPO_HPP
#define RAI_PPO_HPP

#include <iostream>
#include "glog/logging.h"

#include "rai/tasks/common/Task.hpp"
#include <Eigen/Core>
#include <rai/noiseModel/NormalDistributionNoise.hpp>
#include <rai/noiseModel/NoNoise.hpp>
#include <Eigen/Cholesky>
#include <Eigen/Jacobi>
#include <Eigen/Cholesky>
#include <boost/bind.hpp>
#include <math.h>
#include "rai/RAI_core"
#include <vector>
#include <rai/common/math/RAI_math.hpp>

// Neural network
//function approximations
#include "rai/function/common/Policy.hpp"
#include "rai/function/common/ValueFunction.hpp"
#include "rai/function/common/StochasticPolicy.hpp"
#include "rai/common/VectorHelper.hpp"

// memory
#include "rai/memory/Trajectory.hpp"

// acquisitor
#include "rai/experienceAcquisitor/TrajectoryAcquisitor_MultiThreadBatch.hpp"
#include <rai/experienceAcquisitor/TrajectoryAcquisitor_SingleThreadBatch.hpp>
#include <rai/experienceAcquisitor/TrajectoryAcquisitor_Sequential.hpp>
#include <rai/algorithm/common/LearningData.hpp>

// common
#include "rai/common/enumeration.hpp"
#include "rai/common/math/inverseUsingCholesky.hpp"
#include "rai/common/math/ConjugateGradient.hpp"
#include "math.h"
#include "rai/RAI_core"
#include "common/PerformanceTester.hpp"

namespace rai {
namespace Algorithm {

template<typename Dtype, int StateDim, int ActionDim>
class PPO {

 public:

  typedef Eigen::Matrix<Dtype, StateDim, 1> State;
  typedef Eigen::Matrix<Dtype, StateDim, Eigen::Dynamic> StateBatch;
  typedef Eigen::Matrix<Dtype, ActionDim, 1> Action;
  typedef Eigen::Matrix<Dtype, 1, 1> Value;
  typedef Eigen::Matrix<Dtype, 1, Eigen::Dynamic> ValueBatch;
  typedef Eigen::Matrix<Dtype, ActionDim, ActionDim> Covariance;
  typedef Eigen::Matrix<Dtype, -1, -1> MatrixXD;
  typedef Eigen::Matrix<Dtype, -1, 1> VectorXD;
  typedef Eigen::Matrix<Dtype, -1, 1> Parameter;

  using Task_ = Task::Task<Dtype, StateDim, ActionDim, 0>;
  using Noise_ = Noise::NormalDistributionNoise<Dtype, ActionDim>;
  using Policy_ = FuncApprox::StochasticPolicy<Dtype, StateDim, ActionDim>;
  using Trajectory_ = Memory::Trajectory<Dtype, StateDim, ActionDim>;
  using Acquisitor_ = ExpAcq::TrajectoryAcquisitor<Dtype, StateDim, ActionDim>;
  using ValueFunc_ = FuncApprox::ValueFunction<Dtype, StateDim>;

  PPO(std::vector<Task_ *> &tasks,
      ValueFunc_ *vfunction,
      Policy_ *policy,
      std::vector<Noise_ *> &noises,
      Acquisitor_ *acquisitor,
      Dtype lambda,
      int K,
      int numofjunctions,
      unsigned testingTrajN,
      int n_epoch = 30, bool KL_adapt = true,
      Dtype Cov = 1, Dtype Clip_param = 0.2, Dtype Ent_coeff = 0.01,
      Dtype KL_thres = 0.01, Dtype KL_coeff = 1) :
      task_(tasks),
      vfunction_(vfunction),
      policy_(policy),
      noise_(noises),
      acquisitor_(acquisitor),
      lambda_(lambda),
      numOfBranchPerJunct_(K),
      numOfJunct_(numofjunctions),
      testingTrajN_(testingTrajN),
      KL_adapt_(KL_adapt),
      n_epoch_(n_epoch),
      cov_in(Cov),
      KL_thres_(KL_thres),
      KL_coeff_(KL_coeff),
      clip_param_(Clip_param),
      Ent_coeff_(Ent_coeff),
      ld_(acquisitor){

    Utils::logger->addVariableToLog(2, "klD", "");
    Utils::logger->addVariableToLog(2, "Stdev", "");
    Utils::logger->addVariableToLog(2, "klcoef", "");

    parameter_.setZero(policy_->getLPSize());
    policy_->getLP(parameter_);
    policy_->setPPOparams(KL_coeff_, Ent_coeff, Clip_param);

    termCost = task_[0]->termValue();
    discFactor = task_[0]->discountFtr();
    dt = task_[0]->dt();
    timeLimit = task_[0]->timeLimit();
    for (int i = 0; i < task_.size(); i++)
      noiseBasePtr_.push_back(noise_[i]);

    ///update input stdev
    stdev_o.setOnes();
    stdev_o *= std::sqrt(cov_in);
    policy_->setStdev(stdev_o);
    updatePolicyVar();
  };

  ~PPO() {};

  void runOneLoop(int numOfSteps) {
    iterNumber_++;
    tester_.testPerformance(task_,
                            noiseBasePtr_,
                            policy_,
                            task_[0]->timeLimit(),
                            testingTrajN_,
                            ld_.stepsTaken(),
                            vis_lv_,
                            std::to_string(iterNumber_));
    LOG(INFO) << "Simulation";
    ld_.acquireVineTrajForNTimeSteps(task_,
                                     noiseBasePtr_,
                                     policy_,
                                     numOfSteps,
                                     numOfJunct_,
                                     numOfBranchPerJunct_,
                                     vfunction_,
                                     vis_lv_);

    LOG(INFO) << "PPO Updater";
    PPOUpdater();
  }

  void setVisualizationLevel(int vis_lv) { vis_lv_ = vis_lv; }

 private:

  void PPOUpdater() {

    Utils::timer->startTimer("policy Training");
    Utils::timer->startTimer("GAE");

    /// Update Advantage
    advantage_.resize(ld_.stateBat.cols());
    bellmanErr_.resize(ld_.stateBat.cols());
    ValueBatch valuePred(ld_.stateBat.cols());
    Dtype loss;
    LOG(INFO) << "Optimizing policy";


    int dataID = 0;
    for (auto &tra : ld_.traj) {
      ValueBatch advTra = tra.getGAE(vfunction_, discFactor, lambda_, termCost);
      advantage_.block(0, dataID, 1, advTra.cols()) = advTra;
      bellmanErr_.block(0, dataID, 1, advTra.cols()) = tra.bellmanErr;
      dataID += advTra.cols();
    }
    Eigen::Matrix<Dtype,-1,1> temp;
    rai::Math::MathFunc::normalize(advantage_);

    Utils::timer->stopTimer("GAE");

    /// Update Policy & Value
    // TODO : Apply minibatch to mlp and rnn. To train rnn, we need to keep inner states for each minibatches.
    Parameter policy_grad = Parameter::Zero(policy_->getLPSize());
    Dtype KL = 0, KLsum = 0;
    vfunction_->forward(ld_.stateBat, valuePred);

    ValueBatch testt;
    testt.setZero();
    vfunction_->forward(ld_.stateBat,testt);

    for (int i = 0; i < n_epoch_; i++) {

      Utils::timer->startTimer("Vfunction update");

      loss = vfunction_->performOneSolverIter_trustregion(ld_.stateBat, ld_.valueBat, valuePred);

      Utils::timer->stopTimer("Vfunction update");

      policy_->getStdev(stdev_o);
      LOG_IF(FATAL, isnan(stdev_o.norm())) << "stdev is nan!" << stdev_o.transpose();
      Utils::timer->startTimer("Gradient computation");
      if (KL_adapt_) {
          policy_->PPOpg_kladapt(ld_.stateTensor,
                                 ld_.actionTensor,
                                 ld_.actionNoiseTensor,
                                 advantage_,
                                 stdev_o,
                                 ld_.trajLength,
                                 policy_grad);
      } else {
          policy_->PPOpg(ld_.stateTensor,
                         ld_.actionTensor,
                         ld_.actionNoiseTensor,
                         advantage_,
                         stdev_o,
                         ld_.trajLength,
                         policy_grad);
      }

      Utils::timer->stopTimer("Gradient computation");

      LOG_IF(FATAL, isnan(policy_grad.norm())) << "policy_grad is nan!" << policy_grad.transpose();

      Utils::timer->startTimer("Adam update");
      policy_->trainUsingGrad(policy_grad);
      Utils::timer->stopTimer("Adam update");

      KL = policy_->PPOgetkl(ld_.stateTensor, ld_.actionTensor, ld_.actionNoiseTensor, stdev_o, ld_.trajLength);

      LOG_IF(FATAL, isnan(KL)) << "KL is nan!" << KL;
      KLsum += KL;

      if (KL_adapt_) {
        if (KL > KL_thres_ * 1.5)
          KL_coeff_ *= 2;
        if (KL < KL_thres_ / 1.5)
          KL_coeff_ *= 0.5;

        policy_->setPPOparams(KL_coeff_, Ent_coeff_, clip_param_);
      }
    }
    updatePolicyVar();/// save stdev & Update Noise Covariance
    Utils::timer->stopTimer("policy Training");

///Logging
    LOG(INFO) << "Mean KL divergence = " << KLsum / n_epoch_;
    if (KL_adapt_) LOG(INFO) << "KL coefficient = " << KL_coeff_;

    Utils::logger->appendData("Stdev", ld_.stepsTaken(), policy_grad.norm());
    Utils::logger->appendData("klcoef", ld_.stepsTaken(), KL_coeff_);
    Utils::logger->appendData("klD", ld_.stepsTaken(), KLsum / n_epoch_);
  }

  void updatePolicyVar() {
    Action temp;
    policy_->getStdev(stdev_o);
    temp = stdev_o;
    temp = temp.array().square(); //var
    policycov = temp.asDiagonal();
    for (auto &noise : noise_)
      noise->updateCovariance(policycov);
  }

  /////////////////////////// Core //////////////////////////////////////////
  std::vector<Task_ *> task_;
  std::vector<Noise_ *> noise_;
  std::vector<Noise::Noise<Dtype, ActionDim> *> noiseBasePtr_;
  ValueFunc_ *vfunction_;
  Policy_ *policy_;
  Acquisitor_ *acquisitor_;
  Dtype lambda_;
  PerformanceTester<Dtype, StateDim, ActionDim> tester_;
  LearningData<Dtype, StateDim, ActionDim> ld_;

  /////////////////////////// Algorithmic parameter ///////////////////
  int numOfJunct_;
  int numOfBranchPerJunct_;
  int n_epoch_;
  Dtype cov_in;
  Dtype termCost;
  Dtype discFactor;
  Dtype dt;
  Dtype clip_param_;
  Dtype Ent_coeff_;
  Dtype KL_coeff_;
  Dtype KL_thres_;
  double timeLimit;
  bool KL_adapt_;

  /////////////////////////// batches
  ValueBatch advantage_ , bellmanErr_;

  /////////////////////////// Policy parameter
  VectorXD parameter_;
  Action stdev_o;
  Covariance policycov;

  /////////////////////////// plotting
  int iterNumber_ = 0;

  ///////////////////////////testing
  unsigned testingTrajN_;

  /////////////////////////// visualization
  int vis_lv_ = 0;
};

}
}
#endif //RAI_PPO_HPP
