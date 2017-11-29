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
#include <raiCommon/math/RAI_math.hpp>

// Neural network
//function approximations
#include "rai/function/common/Policy.hpp"
#include "rai/function/common/ValueFunction.hpp"
#include "rai/function/common/StochasticPolicy.hpp"
#include "rai/common/VectorHelper.hpp"

// memory
#include "rai/memory/Trajectory.hpp"

// acquisitor
#include "rai/experienceAcquisitor/TrajectoryAcquisitor_Parallel.hpp"
#include <rai/experienceAcquisitor/TrajectoryAcquisitor_Sequential.hpp>
#include <rai/algorithm/common/LearningData.hpp>

// common
#include "raiCommon/enumeration.hpp"
#include "raiCommon/math/inverseUsingCholesky.hpp"
#include "raiCommon/math/ConjugateGradient.hpp"
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
  typedef rai::Algorithm::LearningData<Dtype, StateDim, ActionDim> Dataset;

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
      int numOfBranchPerJunction,
      int numofJunctions,
      unsigned testingTrajN,
      int n_epoch = 30,
      int minibatchSize = 0,
      bool KL_adapt = true,
      Dtype Cov = 1, Dtype Clip_param = 0.2, Dtype Ent_coeff = 0.01,
      Dtype KL_thres = 0.01, Dtype KL_coeff = 1) :
      task_(tasks),
      vfunction_(vfunction),
      policy_(policy),
      noise_(noises),
      acquisitor_(acquisitor),
      lambda_(lambda),
      numOfBranchPerJunct_(numOfBranchPerJunction),
      numOfJunct_(numofJunctions),
      testingTrajN_(testingTrajN),
      KL_adapt_(KL_adapt),
      n_epoch_(n_epoch),
      minibatchSize_(minibatchSize),
      cov_in(Cov),
      KL_thres_(KL_thres),
      KL_coeff_(KL_coeff),
      clip_param_(Clip_param),
      Ent_coeff_(Ent_coeff), Dataset_() {

    ///Construct Dataset
    acquisitor_->setData(&Dataset_);
    Dataset_.miniBatch = new Dataset;

    ///Additional valueTensor for Trustregion update
    //// Tensor
    Tensor<Dtype, 2> valuePred("predictedValue");
    Dataset_.append(valuePred);

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

  ~PPO() { delete Dataset_.miniBatch; };

  void runOneLoop(int numOfSteps) {
    iterNumber_++;
    tester_.testPerformance(task_,
                            noiseBasePtr_,
                            policy_,
                            task_[0]->timeLimit(),
                            testingTrajN_,
                            acquisitor_->stepsTaken(),
                            vis_lv_,
                            std::to_string(iterNumber_));
    LOG(INFO) << "Simulation";
    acquisitor_->acquireVineTrajForNTimeSteps(task_,
                                              noiseBasePtr_,
                                              policy_,
                                              numOfSteps,
                                              numOfJunct_,
                                              numOfBranchPerJunct_,
                                              vfunction_,
                                              vis_lv_);
    acquisitor_->saveData(task_[0], policy_, vfunction_);

    LOG(INFO) << "PPO Updater";
    PPOUpdater();
  }

  void setVisualizationLevel(int vis_lv) { vis_lv_ = vis_lv; }

 private:

  void PPOUpdater() {

    Utils::timer->startTimer("policy Training");

    Utils::timer->startTimer("GAE");
    acquisitor_->computeAdvantage(task_[0], vfunction_, lambda_);
    Utils::timer->stopTimer("GAE");

    Dtype loss;
    LOG(INFO) << "Optimizing policy";

    /// Update Policy & Value
    Parameter policy_grad = Parameter::Zero(policy_->getLPSize());
    Dtype KL = 0, KLsum = 0;
    int cnt = 0;

    /// Append predicted value to Dataset_ for trust region update
    Dataset_.extraTensor2D[0].resize(Dataset_.maxLen, Dataset_.batchNum);

    vfunction_->forward(Dataset_.states, Dataset_.extraTensor2D[0]);

//    ///Visualize Data
//    rai::Tensor<Dtype,2> testv;
//    rai::Tensor<Dtype,2> v_plot;
//    MatrixXD state_plot0, state_plot1, v_plot0;
//    state_plot0.resize(1, Dataset_.maxLen * Dataset_.batchNum);
//    state_plot1.resize(1, Dataset_.maxLen * Dataset_.batchNum);
//    v_plot0.resize(1, Dataset_.maxLen * Dataset_.batchNum);
//
//    int colID = 0;
//    for (int i = 0; i < Dataset_.batchNum; i++) {
//      for (int t = 0; t <  Dataset_.maxLen ; t++) {
//        v_plot0(colID) = Dataset_.values.eMat()(t,i);
//        state_plot0(colID) = Dataset_.states.eTensor()(0,t,i);
//        state_plot1(colID++) = t;
//      }
//    }
//    rai::Utils::Graph::FigProp3D figprop;
//    figprop.title = "V function";
//    figprop.xlabel = "angle";
//    figprop.ylabel = "T";
//    figprop.zlabel = "value";
//    figprop.displayType = rai::Utils::Graph::DisplayType3D::heatMap3D;
//
//
//    Utils::graph->figure3D(4, figprop);
//    Utils::graph->append3D_Data(4, state_plot0.data(), state_plot1.data(), v_plot0.data(), v_plot0.cols(), false, Utils::Graph::PlotMethods3D::points, "groundtruth");
//    Utils::graph->drawFigure(4);
//
//    graph -> waitForEnter();
    std::cout << "dataTest"<<std::endl;
    std::cout << Dataset_.values.row(0) << std::endl;
    std::cout << Dataset_.extraTensor2D[0].row(0) << std::endl << std::endl;
    for (int i = 0; i < n_epoch_; i++) {
      while (Dataset_.iterateBatch(minibatchSize_)) {
        Utils::timer->startTimer("Vfunction update");
        if (vfunction_->isRecurrent())
          loss = vfunction_->performOneSolverIter_trustregion(Dataset_.miniBatch->states,
                                                              Dataset_.miniBatch->values,
                                                              Dataset_.miniBatch->extraTensor2D[0],Dataset_.miniBatch->lengths);
        else
        loss = vfunction_->performOneSolverIter_trustregion(Dataset_.miniBatch->states,
                                                            Dataset_.miniBatch->values,
                                                            Dataset_.miniBatch->extraTensor2D[0]);
        Utils::timer->stopTimer("Vfunction update");


//        testv.resize(Dataset_.maxLen, Dataset_.miniBatch->batchNum);
//
//        vfunction_->forward(Dataset_.miniBatch->states,testv);
//
//        Eigen::Matrix<Dtype, -1, -1 > test1, test2, test3;
//        test1 = Dataset_.miniBatch->values.eMat(); // target
//        test2= testv.eMat();
//        test3 = (test1 - test2).array().square();
//
//        LOG(INFO)  << 0.5 * test3.mean()<< std::endl;
//        LOG(INFO) << loss;




        policy_->getStdev(stdev_o);
        LOG_IF(FATAL, isnan(stdev_o.norm())) << "stdev is nan!" << stdev_o.transpose();
        Utils::timer->startTimer("Gradient computation");
        if (KL_adapt_) policy_->PPOpg_kladapt(Dataset_.miniBatch, stdev_o, policy_grad);
        else policy_->PPOpg(Dataset_.miniBatch, stdev_o, policy_grad);
        Utils::timer->stopTimer("Gradient computation");
        LOG_IF(FATAL, isnan(policy_grad.norm())) << "policy_grad is nan!" << policy_grad.transpose();

        Utils::timer->startTimer("Adam update");
        policy_->trainUsingGrad(policy_grad);
        Utils::timer->stopTimer("Adam update");

        KL = policy_->PPOgetkl(Dataset_.miniBatch, stdev_o);
        LOG_IF(FATAL, isnan(KL)) << "KL is nan!" << KL;

        KLsum += KL;
        cnt++;
      }
      if (KL_adapt_) {
        if (KL > KL_thres_ * 1.5)
          KL_coeff_ *= 2;
        if (KL < KL_thres_ / 1.5)
          KL_coeff_ *= 0.5;

        policy_->setPPOparams(KL_coeff_, Ent_coeff_, clip_param_);
      }
    }
    KL = KLsum / cnt;

    updatePolicyVar();/// save stdev & Update Noise Covariance
    Utils::timer->stopTimer("policy Training");

///Logging
    LOG(INFO) << "Mean KL divergence per epoch = " << KL;
    if (KL_adapt_) LOG(INFO) << "KL coefficient = " << KL_coeff_;

    Utils::logger->appendData("Stdev", acquisitor_->stepsTaken(), policy_grad.norm());
    Utils::logger->appendData("klcoef", acquisitor_->stepsTaken(), KL_coeff_);
    Utils::logger->appendData("klD", acquisitor_->stepsTaken(), KLsum / n_epoch_);
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
  Dataset Dataset_;

  /////////////////////////// Algorithmic parameter ///////////////////
  int numOfJunct_;
  int numOfBranchPerJunct_;
  int n_epoch_;
  int minibatchSize_;
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
