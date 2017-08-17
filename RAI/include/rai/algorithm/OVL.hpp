//
// Created by jhwangbo on 13.04.17.
//

#ifndef RAI_OVL_HPP
#define RAI_OVL_HPP

#include <iostream>

#include "rai/tasks/common/Task.hpp"
#include "rai/memory/ReplayMemorySARS.hpp"
#include "rai/memory/Trajectory.hpp"
#include <Eigen/Core>
#include <rai/noiseModel/OrnsteinUhlenbeckNoise.hpp>
#include <rai/noiseModel/NoNoise.hpp>
#include <boost/bind.hpp>
#include <rai/RAI_core>
#include <rai/experienceAcquisitor/TrajectoryAcquisitor_SingleThreadBatch.hpp>
#include <rai/experienceAcquisitor/TrajectoryAcquisitor_Sequential.hpp>
#include <math/RAI_math.hpp>
#include <rai/noiseModel/NormalDistributionNoise.hpp>
#include "rai/experienceAcquisitor/TrajectoryAcquisitor_MultiThreadBatch.hpp"
#include "rai/memory/BellmanTuples.hpp"

// Neural network
//function approximations
#include "rai/function/common/Policy.hpp"
#include "rai/function/common/ValueFunction.hpp"

// common
#include "enumeration.hpp"
#include "math/inverseUsingCholesky.hpp"
#include "math/ConjugateGradient.hpp"
#include "math.h"
#include "rai/RAI_core"

namespace RAI {
namespace Algorithm {

template<typename Dtype, int StateDim, int ActionDim>
class OVL {

  class BellmanTuple {
   public:
    BellmanTuple(unsigned trajId,
                 unsigned startTimeId,
                 unsigned endTimeId,
                 Dtype discCosts,
                 Dtype cumulDiscFctr,
                 bool terminated) :
        trajId_(trajId),
        startTimeId_(startTimeId),
        endTimeId_(endTimeId),
        discCosts_(discCosts),
        cumulDiscFctr_(cumulDiscFctr),
        terminated_(terminated) {}
    unsigned trajId_, startTimeId_, endTimeId_;
    Dtype discCosts_;
    Dtype cumulDiscFctr_;
    bool terminated_;
  };

 public:
  typedef Eigen::Matrix<Dtype, StateDim, 1> State;
  typedef Eigen::Matrix<Dtype, StateDim, Eigen::Dynamic> StateBatch;
  typedef Eigen::Matrix<Dtype, ActionDim, 1> Action;
  typedef Eigen::Matrix<Dtype, ActionDim, Eigen::Dynamic> ActionBatch;
  typedef Eigen::Matrix<Dtype, 1, Eigen::Dynamic> ValueBatch;
  typedef Eigen::Matrix<Dtype, 1, Eigen::Dynamic> CostBatch;
  typedef Eigen::Matrix<Dtype, StateDim, ActionDim> JacobianStateResAct;
  typedef Eigen::Matrix<Dtype, 1, ActionDim> JacobianCostResAct;
  typedef Eigen::Matrix<Dtype, -1, -1> MatrixXD;
  typedef Eigen::Matrix<Dtype, -1, 1> VectorXD;
  typedef Eigen::Matrix<Dtype, 1, -1> RowVectorXD;

  using Task_ = Task::Task<Dtype, StateDim, ActionDim, 0>;
  using ReplayMemory_ = RAI::Memory::ReplayMemorySARS<Dtype, StateDim, ActionDim>;
  using Qfunction_ = FuncApprox::Qfunction<Dtype, StateDim, ActionDim>;
  using Vfunction_ = FuncApprox::ValueFunction<Dtype, StateDim>;
  using Policy_ = FuncApprox::Policy<Dtype, StateDim, ActionDim>;
  using Noise_ = Noise::Noise<Dtype, ActionDim>;
  using Trajectory_ = Memory::Trajectory<Dtype, StateDim, ActionDim>;
  using Acquisitor_ = ExpAcq::TrajectoryAcquisitor<Dtype, StateDim, ActionDim>;
  using TestAcquisitor_ = ExpAcq::TrajectoryAcquisitor_MultiThreadBatch<Dtype, StateDim, ActionDim>;
//  using TestAcquisitor_ = ExpAcq::TrajectoryAcquisitor_Sequential<Dtype, StateDim, ActionDim>;
// using TestAcquisitor_ = ExpAcq::TrajectoryAcquisitor_SingleThreadBatch<Dtype, StateDim, ActionDim>;

  OVL(std::vector<Task_ *> &tasks,
      Vfunction_ *vfunction,
      Vfunction_ *vfunction_target,
      Qfunction_ *qfunction,
      Policy_ *policy,
      std::vector<Noise_ *> &noise,
      Acquisitor_ *acquisitor,
      unsigned nu,
      Dtype trajTimeLimit,
      unsigned innerLoopN,
      unsigned testingTrajN = 1,
      Dtype pruningPercent = 0.001,
      Dtype minimumTupleSetSize = 1e4) :
      vfunction_(vfunction),
      vfunction_targ_(vfunction_target),
      qfunction_(qfunction),
      policy_(policy),
      noise_(noise),
      acquisitor_(acquisitor),
      testTraj_(testingTrajN),
      nu_(nu),
      trajTimeLimit_(trajTimeLimit),
      innerLoopN_(innerLoopN),
      task_(tasks),
      testingTrajN_(testingTrajN),
      pruningPercent_(pruningPercent),
      minimumTupleSetSize_(minimumTupleSetSize) {
    vfunction_targ_->copyAPFrom(vfunction);
    discountFctr_ = task_[0]->discountFtr();
    termValue_ = task_[0]->termValue();
    Utils::logger->addVariableToLog(2, "Nominal performance", "");
  }

  void initiallyFillTheMemory(unsigned nofTraj) {
    std::vector<Trajectory_> traj(nofTraj);
    StateBatch startState(StateDim, nofTraj);
    sampleBatchOfInitial(startState);
    if (vis_lv_ > 1) task_[0]->turnOnVisualization("");
    Utils::timer->startTimer("Simulation");
    acquisitor_->acquire(task_, policy_, noise_, traj, startState, trajTimeLimit_, true);
    Utils::timer->stopTimer("Simulation");
    if (vis_lv_ > 1) task_[0]->turnOffVisualization();
    appendTraj(traj, nu_, discountFctr_);
  }

  void learn4Ntraj(unsigned nofTraj) {

    //////////////// testing (not part of the algorithm) ////////////////////
    Utils::timer->disable();

    for (auto &tra : testTraj_)
      tra.clear();
    for (auto &noise : noise_)
      noise->initializeNoise();
    for (auto &task : task_)
      task->setToInitialState();

    StateBatch startState(StateDim, testingTrajN_);
    sampleBatchOfInitial(startState);

    Noise::NoNoise<Dtype, ActionDim> noNoises[task_.size()];
    std::vector<Noise_ *> noiseVec;
    for (unsigned i = 0; i < task_.size(); i++)
      noiseVec.push_back(&noNoises[i]);

    if (vis_lv_ > 0) task_[0]->turnOnVisualization("");
    Dtype averageCost = testAcquisitor_.acquire(task_,
                                                policy_,
                                                noiseVec,
                                                testTraj_,
                                                startState,
                                                task_[0]->timeLimit(),
                                                false);
    if (vis_lv_ > 0) task_[0]->turnOffVisualization();
    Utils::logger->appendData("Nominal performance",
                              float(acquisitor_->stepsTaken()),
                              float(averageCost));
    LOG(INFO) << "steps taken " << logger->getData("Nominal performance")->at(0).back()
              << ", average cost " << logger->getData("Nominal performance")->at(1).back();

    /// reset all for learning
    for (auto &task : task_)
      task->setToInitialState();
    for (auto &noise : noise_)
      noise->initializeNoise();

    Utils::timer->enable();
    /////////////////////////////////////////////////////////////////////////

    learnForOneIteration(nofTraj);
  }

  void setVisualizationLevel(int vis_lv) { vis_lv_ = vis_lv; }

 private:

  void learnForOneIteration(unsigned nOfTraj) {

    std::vector<Trajectory_> traj(nOfTraj);
    StateBatch startState(StateDim, nOfTraj);
    sampleRandStates(tuplesV_, startState);
    if (vis_lv_ > 1) task_[0]->turnOnVisualization("");
    Utils::timer->startTimer("Simulation");
    acquisitor_->acquire(task_, policy_, noise_, traj, startState, trajTimeLimit_, true);
    Utils::timer->stopTimer("Simulation");
    if (vis_lv_ > 1) task_[0]->turnOffVisualization();
    appendTraj(traj, nu_, discountFctr_);

    for (int i = 0; i < innerLoopN_ + 1; i++) {
      updateV();
      updateQandP();
    }
  }

  void updateV() {
    unsigned batchSize = tuplesV_.size() / 2;
    StateBatch state_t_Batch(StateDim, batchSize);
    ActionBatch action_t_Batch(ActionDim, batchSize);
    StateBatch state_tk_Batch(StateDim, batchSize);
    CostBatch cumCost_Batch(1, batchSize);
    CostBatch dicFtr_Batch(1, batchSize);
    CostBatch termValue_Batch(1, batchSize);
    CostBatch value_Batch(1, batchSize);
    CostBatch valueEstV_Batch(1, batchSize);
    std::vector<bool> termflag(batchSize);
    std::vector<unsigned> idx;
    /////////////plotting ///////////////////
    ///////////////////////// plotting for debugging ///////////////////////////////
    StateBatch state_plot(3, 2601);
    ActionBatch action_plot(1, 2601);
    CostBatch value_plot(1, 2601);
    MatrixXD minimal_X_extended(1, 2601);
    MatrixXD minimal_Y_extended(1, 2601);

    MatrixXD minimal_X_sampled(1, 2601);
    MatrixXD minimal_Y_sampled(1, 2601);
    ActionBatch action_sampled(1, 2601);

    for (int i = 0; i < 51; i++) {
      for (int j = 0; j < 51; j++) {
        minimal_X_extended(i * 51 + j) = -M_PI + M_PI * i / 25.0;
        minimal_Y_extended(i * 51 + j) = -5.0 + j / 25.0 * 5.0;
        state_plot(0, i * 51 + j) = cos(minimal_X_extended(i * 51 + j));
        state_plot(1, i * 51 + j) = sin(minimal_X_extended(i * 51 + j));
        state_plot(2, i * 51 + j) = minimal_Y_extended(i * 51 + j);
      }
    }
    RAI::Utils::Graph::FigProp3D figurePropertiesSVC;
    figurePropertiesSVC.title = "V function";
    figurePropertiesSVC.xlabel = "angle";
    figurePropertiesSVC.ylabel = "angular velocity";
    figurePropertiesSVC.zlabel = "value";
    figurePropertiesSVC.displayType = RAI::Utils::Graph::DisplayType3D::heatMap3D;
    CostBatch xDataplot(1, batchSize);
    CostBatch yDataplot(1, batchSize);
    CostBatch zDataplot(1, batchSize);
    /////////////////////////////////////////
    Utils::timer->startTimer("value func learning");
    sampleRandBellTuples(tuplesV_, state_t_Batch,
                         state_tk_Batch,
                         action_t_Batch,
                         cumCost_Batch,
                         dicFtr_Batch,
                         termflag,
                         idx);
    vfunction_targ_->forward(state_tk_Batch, termValue_Batch);
    for (unsigned k = 0; k < batchSize; k++) {
      if (termflag[k]) termValue_Batch(k) = termValue_;
      value_Batch(k) = cumCost_Batch(k) + dicFtr_Batch(k) * termValue_Batch(k);
    }

    vfunction_->performOneSolverIter_infimum(state_t_Batch, value_Batch, 1e-4);
    vfunction_targ_->interpolateAPWith(vfunction_, 0.03);
    Utils::timer->stopTimer("value func learning");


    /// pruning v tuples
    Utils::timer->startTimer("value func pruning");
    vfunction_->forward(state_t_Batch, valueEstV_Batch);
    CostBatch disadvantageV = value_Batch - valueEstV_Batch;
    Dtype std = RAI::Math::MathFunc::standardDev(disadvantageV);
    Dtype mean = disadvantageV.mean();
    std::vector<unsigned> idx4pruning;

    for (unsigned k = 0; k < batchSize; k++)
      if (disadvantageV(k) - mean > Dtype(2.5) * std) idx4pruning.push_back(idx[k]);
//    popTuples(tuplesV_, idx4pruning);
    Utils::timer->stopTimer("value func pruning");

    for(unsigned k = 0; k < 1 ; k++)
      policy_->backwardUsingCritic(qfunction_, state_t_Batch);

    /////////plotting
    unsigned optimalOnes = 0;
    for (unsigned k = 0; k < batchSize; k++)
      if ( disadvantageV(k) < Dtype(0)) optimalOnes++;

    CostBatch xDataplotPolicy(1, optimalOnes);
    CostBatch yDataplotPolicy(1, optimalOnes);
    CostBatch zDataplotPolicy(1, optimalOnes);
    optimalOnes = 0;

    for (unsigned k=0; k < batchSize; k++){
      xDataplot[k] = std::atan2(state_t_Batch(1,k), state_t_Batch(0,k));
      yDataplot[k] = state_t_Batch(2,k);
      zDataplot[k] = value_Batch(k);
      if ( disadvantageV(k) < Dtype(0)) {
        xDataplotPolicy[optimalOnes] = std::atan2(state_t_Batch(1,k), state_t_Batch(0,k));
        yDataplotPolicy[optimalOnes] = state_t_Batch(2,k);
        zDataplotPolicy[optimalOnes++] = action_t_Batch(k);
      }
    }
    vfunction_->forward(state_plot, value_plot);
    graph->drawHeatMap(3, figurePropertiesSVC, minimal_X_extended.data(),
                       minimal_Y_extended.data(), value_plot.data(), 51, 51, "");
    graph->append3D_Data(3, xDataplot.data(), yDataplot.data(), zDataplot.data(), batchSize, false, Utils::Graph::PlotMethods3D::points, "","pt 7");
    graph->drawFigure(3);

//    graph->figure3D(5, figurePropertiesSVC);
//    graph->append3D_Data(5, xDataplotPolicy.data(), yDataplotPolicy.data(), zDataplotPolicy.data(), optimalOnes, false, Utils::Graph::PlotMethods3D::points, "","");
//    graph->drawFigure(5);
  }

  void updateQandP() {
    unsigned batchSize = tuplesQ_.size() / 2;
    StateBatch state_t_Batch(StateDim, batchSize);
    ActionBatch action_t_Batch(ActionDim, batchSize);
    StateBatch state_tk_Batch(StateDim, batchSize);
    CostBatch cumCost_Batch(1, batchSize);
    CostBatch dicFtr_Batch(1, batchSize);
    CostBatch termValue_Batch(1, batchSize);
    CostBatch value_Batch(1, batchSize);
    CostBatch valueEstQ_Batch(1, batchSize);
    std::vector<bool> termflag(batchSize);
    std::vector<unsigned> idx;
//    /////////////plotting ///////////////////
//    ///////////////////////// plotting for debugging ///////////////////////////////
//    StateBatch state_plot(1, 2601);
//    ActionBatch action_plot(1, 2601);
//    CostBatch value_plot(1, 2601);
//    ActionBatch action_sampled(1, 51);
//
//    for (int i = 0; i < 51; i++) {
//      for (int j = 0; j < 51; j++) {
//        state_plot(i * 51 + j) = -1 + 1 * i / 25.0;
//        action_plot(i * 51 + j) = -5.0 + j / 25.0 * 5.0;
//      }
//    }
//    RAI::Utils::Graph::FigProp3D figurePropertiesSVC;
//    figurePropertiesSVC.title = "Q function";
//    figurePropertiesSVC.xlabel = "state";
//    figurePropertiesSVC.ylabel = "action";
//    figurePropertiesSVC.zlabel = "value";
//    figurePropertiesSVC.displayType = RAI::Utils::Graph::DisplayType3D::heatMap3D;
////////////////////////////////////////////////////////////////////////////////////////////

    Utils::timer->startTimer("q func learning");
    sampleRandBellTuples(tuplesQ_, state_t_Batch,
                         state_tk_Batch,
                         action_t_Batch,
                         cumCost_Batch,
                         dicFtr_Batch,
                         termflag,
                         idx);
    vfunction_targ_->forward(state_tk_Batch, termValue_Batch);
    for (unsigned k = 0; k < batchSize; k++) {
      if (termflag[k]) termValue_Batch(k) = termValue_;
      value_Batch(k) = cumCost_Batch(k) + dicFtr_Batch(k) * termValue_Batch(k);
    }


    qfunction_->performOneSolverIter_infimum(state_t_Batch, action_t_Batch, value_Batch, 1e-4);
    Utils::timer->stopTimer("q func learning");

    /// pruning q tuples
    Utils::timer->startTimer("q func pruning");

    qfunction_->forward(state_t_Batch, action_t_Batch, valueEstQ_Batch);
    CostBatch disadvantageQ = value_Batch - valueEstQ_Batch;
    Dtype std = RAI::Math::MathFunc::standardDev(valueEstQ_Batch);
    Dtype mean = valueEstQ_Batch.mean();

    std::vector<unsigned> idx4pruning;

    for (unsigned k = 0; k < batchSize; k++)
      if( valueEstQ_Batch(k) - mean > Dtype(2.5) * std ) idx4pruning.push_back(idx[k]);

//    popTuples(tuplesQ_, idx4pruning);
    Utils::timer->stopTimer("q func pruning");

//    /////////plotting
//    qfunction_->forward(state_plot, action_plot, value_plot);
//    graph->drawHeatMap(3, figurePropertiesSVC, state_plot.data(),
//                       action_plot.data(), value_plot.data(), 51, 51, "");
//    graph->append3D_Data(3, state_t_Batch.data(), action_t_Batch.data(), value_Batch.data(), batchSize, false, Utils::Graph::PlotMethods3D::points, "","");
//    graph->drawFigure(3);
  }

  void sampleBatchOfInitial(StateBatch &initialBatch) {
    for (unsigned trajID = 0; trajID < initialBatch.cols(); trajID++) {
      State state;
      task_[0]->setToInitialState();
      task_[0]->getState(state);
      initialBatch.col(trajID) = state;
    }
  }

  void sampleRandBellTuples(std::vector<BellmanTuple> &tuple,
                            StateBatch &state_t_batch,
                            StateBatch &state_tk_batch,
                            ActionBatch &action_t_batch,
                            CostBatch &discCost_batch,
                            CostBatch &dictFctr_batch,
                            std::vector<bool> &terminalState,
                            std::vector<unsigned> &idx) {
    unsigned batchSize = state_t_batch.cols();
    unsigned size = tuple.size();
    LOG_IF(FATAL, tuple.size() < state_t_batch.cols() * 1.2) <<
                                                             "You don't have enough memories in the storage! accumulate more memories before you update your Q-function";
    idx = rn_.getNrandomSubsetIdx(tuple.size(), state_t_batch.cols());

    ///// saving memory to the batch
    for (unsigned i = 0; i < batchSize; i++) {
      unsigned j = idx[i];
      state_t_batch.col(i) = traj_[tuple[j].trajId_].stateTraj[tuple[j].startTimeId_];
      state_tk_batch.col(i) = traj_[tuple[j].trajId_].stateTraj[tuple[j].endTimeId_];
      action_t_batch.col(i) = traj_[tuple[j].trajId_].actionTraj[tuple[j].startTimeId_];
      discCost_batch[i] = tuple[j].discCosts_;
      dictFctr_batch[i] = tuple[j].cumulDiscFctr_;
      terminalState[i] = tuple[j].terminated_;
    }
  }

  void sampleRandStates(std::vector<BellmanTuple> &tuple, StateBatch &state_t_batch) {
    LOG_IF(FATAL, tuple.size() < state_t_batch.cols() * 1.2) <<
                                                             "You don't have enough memories in the storage! accumulate more memories before you update your Q-function";
    std::vector<unsigned> idx = rn_.getNrandomSubsetIdx(tuple.size(), state_t_batch.cols());
    ///// saving memory to the batch
    for (unsigned i = 0; i < state_t_batch.cols(); i++)
      state_t_batch.col(i) = traj_[tuple[idx[i]].trajId_].stateTraj[tuple[idx[i]].startTimeId_];
  }

  /// nu=1 means only the transition tuples
  void appendTraj(std::vector<Trajectory_> &traj, unsigned nu, Dtype discFtr) {
    unsigned traId = traj_.size();
    for (auto &tra: traj) {
      traj_.push_back(tra);
      for (int i = 0; i < tra.size() - 2; i++) {
        for (int j = 1; j < nu + 1 && i + j < tra.size() - 1; j++) {
          Dtype cumDiscF = pow(discFtr, j);
          Dtype cumCost = Dtype(0);
          for (int k = j - 1; k > -1; k--)
            cumCost = tra.costTraj[i + k] + cumCost * discFtr;

          bool terminated = i + j == tra.size() - 1 && tra.termType == TerminationType::terminalState;
          tuplesQ_.push_back(BellmanTuple(traId, i, i + j, cumCost, cumDiscF, terminated));
          tuplesV_.push_back(BellmanTuple(traId, i, i + j, cumCost, cumDiscF, terminated));
        }
      }
      traId++;
    }
  }

  void popTuples(std::vector<BellmanTuple> &tuple, std::vector<unsigned> &idx) {
    std::vector<BellmanTuple> tuples_copy = tuple;
    tuple.clear();
    bool keep[tuples_copy.size()];
    for (int i = 0; i < tuples_copy.size(); i++)
      keep[i] = true;
    for (int i = 0; i < idx.size(); i++)
      keep[idx[i]] = false;
    for (int i = 0; i < tuples_copy.size(); i++)
      if (keep[i]) tuple.push_back(tuples_copy[i]);
  }

/// core
  Vfunction_ *vfunction_, *vfunction_targ_;
  Qfunction_ *qfunction_;
  Policy_ *policy_;
  Acquisitor_ *acquisitor_;
  std::vector<Noise_ *> noise_;
  std::vector<Task_ *> task_;
  Dtype trajTimeLimit_;
  std::vector<Trajectory_> testTraj_;
  Dtype discountFctr_;
  unsigned nu_;
  unsigned miniBatchSize_;
  unsigned innerLoopN_;
  Dtype termValue_;
  Dtype pruningPercent_;
  Dtype minimumTupleSetSize_;

  std::vector<Trajectory_> traj_;
  std::vector<BellmanTuple> tuplesV_;
  std::vector<BellmanTuple> tuplesQ_;
  RandomNumberGenerator<Dtype> rn_;

/// mics
  int vis_lv_ = 0;
  unsigned testingTrajN_ = 1;
  TestAcquisitor_ testAcquisitor_;

};

}
}
#endif //RAI_OVL_HPP
