//
// Created by jhwangbo on 08/08/17.
// This class's old data is deleted when you acquire new data
//

#ifndef RAI_LearningData_HPP
#define RAI_LearningData_HPP

#include <rai/memory/Trajectory.hpp>
#include <rai/experienceAcquisitor/TrajectoryAcquisitor.hpp>
#include <rai/experienceAcquisitor/ExperienceTupleAcquisitor.hpp>
#include <rai/RAI_core>
#include <rai/noiseModel/NormalDistributionNoise.hpp>
#include <rai/function/common/StochasticPolicy.hpp>
#include <rai/common/VectorHelper.hpp>

namespace rai {
namespace Algorithm {

template<typename Dtype, int StateDim, int ActionDim>
class LearningData {

  typedef Eigen::Matrix<Dtype, StateDim, 1> State;
  typedef Eigen::Matrix<Dtype, StateDim, Eigen::Dynamic> StateBatch;
  typedef Eigen::Matrix<Dtype, ActionDim, 1> Action;
  typedef Eigen::Matrix<Dtype, ActionDim, Eigen::Dynamic> ActionBatch;
  typedef Eigen::Matrix<Dtype, 1, 1> Value;
  typedef Eigen::Matrix<Dtype, 1, Eigen::Dynamic> ValueBatch;
  typedef Eigen::Matrix<Dtype, StateDim, ActionDim> JacobianStateResAct;
  typedef Eigen::Matrix<Dtype, 1, ActionDim> JacobianCostResAct;
  typedef Eigen::Matrix<Dtype, ActionDim, Eigen::Dynamic> JacobianActResParam;
  typedef Eigen::Matrix<Dtype, ActionDim, ActionDim> FimInActionSapce;
  typedef Eigen::Matrix<Dtype, ActionDim, ActionDim> Covariance;
  typedef Eigen::Matrix<Dtype, -1, -1> MatrixXD;
  typedef Eigen::Matrix<Dtype, -1, 1> VectorXD;
  typedef Eigen::Matrix<Dtype, -1, 1> Parameter;

  using Trajectory_ = Memory::Trajectory<Dtype, StateDim, ActionDim>;
  using TrajAcquisitor_ = ExpAcq::TrajectoryAcquisitor<Dtype, StateDim, ActionDim>;
  using TupleAcquisitor_ = ExpAcq::ExperienceTupleAcquisitor<Dtype, StateDim, ActionDim>;
  using Task_ = Task::Task<Dtype, StateDim, ActionDim, 0>;
  using Noise_ = Noise::Noise<Dtype, ActionDim>;
  using Policy_ = FuncApprox::Policy<Dtype, StateDim, ActionDim>;
  using ValueFunc_ = FuncApprox::ValueFunction<Dtype, StateDim>;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  LearningData(TrajAcquisitor_ *acq) : trajAcquisitor_(acq), stateTensor(),
                                       actionTensor("sampled_oa"),
                                       actionNoiseTensor("noise_oa"),
                                       trajLength("length"), advantageTensor("advantage"), cur_ID(0), cur_minibatch(){
  }
  struct tensorBatch{
    tensorBatch():lengths("length"),advantages("advantage"), states("state"), hiddenStates("h_init"), actions("sampled_oa"), actionNoises("noise_oa"){};
    Tensor<Dtype, 1> lengths;
    Tensor<Dtype, 2> advantages;
    Tensor<Dtype, 3> states;
    Tensor<Dtype, 3> hiddenStates;
    Tensor<Dtype, 3> actions;
    Tensor<Dtype, 3> actionNoises;
  };

  void acquireVineTrajForNTimeSteps(std::vector<Task_ *> &task,
                                    std::vector<Noise_ *> &noise,
                                    Policy_ *policy,
                                    int numOfSteps,
                                    int numofjunct,
                                    int numOfBranchPerJunct,
                                    ValueFunc_ *vfunction = nullptr,
                                    int vis_lv = 0) {
    Utils::timer->startTimer("Simulation");

    std::vector<Trajectory_> trajectories;
    double dt = task[0]->dt();
    double timeLimit = task[0]->timeLimit();

    int numOfTra_ = std::ceil(1.1 * numOfSteps * dt / timeLimit);
    traj.resize(numOfTra_);
    StateBatch startState(StateDim, numOfTra_);
    sampleBatchOfInitial(startState, task);
    for (auto &noise : noise)
      noise->initializeNoise();
    for (auto &task : task)
      task->setToInitialState();
    for (auto &tra : traj)
      tra.clear();
    if (vis_lv > 1) task[0]->turnOnVisualization("");
    long double stepsTaken = trajAcquisitor_->stepsTaken();
    Dtype cost = trajAcquisitor_->acquire(task,
                                          policy,
                                          noise,
                                          traj,
                                          startState,
                                          timeLimit,
                                          true);
    if (vis_lv > 1) task[0]->turnOffVisualization();

    int stepsInThisLoop = int(trajAcquisitor_->stepsTaken() - stepsTaken);

    if (numOfSteps > stepsInThisLoop) {
      int stepsneeded = numOfSteps - stepsInThisLoop;
      std::vector<Trajectory_> tempTraj_;
      while (1) {
        int numofnewtraj = std::ceil(1.5 * stepsneeded * dt / timeLimit); // TODO: fix

        tempTraj_.resize(numofnewtraj);
        for (auto &tra : tempTraj_)
          tra.clear();

        StateBatch startState2(StateDim, numofnewtraj);
        sampleBatchOfInitial(startState2, task);

        for (auto &noise : noise)
          noise->initializeNoise();

        if (vis_lv > 1) task[0]->turnOnVisualization("");
        trajAcquisitor_->acquire(task,
                                 policy,
                                 noise,
                                 tempTraj_,
                                 startState2,
                                 timeLimit,
                                 true);
        if (vis_lv > 1) task[0]->turnOffVisualization();

        stepsInThisLoop = int(trajAcquisitor_->stepsTaken() - stepsTaken);
        stepsneeded = numOfSteps - stepsInThisLoop;
        ///merge trajectories
        traj.reserve(traj.size() + tempTraj_.size());
        traj.insert(traj.end(), tempTraj_.begin(), tempTraj_.end());

        if (stepsneeded <= 0) break;
      }
    }
    ///////////////////////////////////////VINE//////////////////////////////
    StateBatch VineStartPosition(StateDim, numofjunct);
    StateBatch rolloutstartState(StateDim, numofjunct * numOfBranchPerJunct);
    trajectories.resize(numofjunct * numOfBranchPerJunct);
    rolloutstartState.setOnes();
    std::vector<std::pair<int, int> > indx;
    rai::Op::VectorHelper::sampleRandomStates(traj, VineStartPosition, int(0.1 * timeLimit / dt), indx);

    for (int dataID = 0; dataID < numofjunct; dataID++)
      rolloutstartState.block(0, dataID * numOfBranchPerJunct, StateDim, numOfBranchPerJunct) =
          rolloutstartState.block(0, dataID * numOfBranchPerJunct, StateDim, numOfBranchPerJunct).array().colwise()
              * VineStartPosition.col(dataID).array();

    for (auto &tra : trajectories)
      tra.clear();
    for (auto &noise : noise)
      noise->initializeNoise();

    trajAcquisitor_->acquire(task, policy, noise, trajectories, rolloutstartState, timeLimit, true);

    ///merge trajectories into one vector
    traj.reserve(traj.size() + trajectories.size());
    traj.insert(traj.end(), trajectories.begin(), trajectories.end());

    processTrajs(task[0], policy, vfunction);
    Utils::timer->stopTimer("Simulation");
  }

  void acquireTrajForNTimeSteps(std::vector<Task_ *> &task,
                                std::vector<Noise_ *> &noise,
                                Policy_ *policy,
                                int numOfSteps,
                                ValueFunc_ *vfunction = nullptr,
                                int vis_lv = 0) {
    acquireVineTrajForNTimeSteps(task, noise, policy, numOfSteps, 0, 0, vfunction, vis_lv);
  }

  long int stepsTaken() {
    long int steps = 0;
    if (trajAcquisitor_)
      steps += trajAcquisitor_->stepsTaken();
    if (tupleAcquisitor_)
      steps += tupleAcquisitor_->stepsTaken();
    return steps;
  }

  void computeAdvantage(Task_ *task, ValueFunc_ *vfunction, Dtype lambda, bool normalize = true) {
    int batchID = 0;
    int dataID = 0;
    advantageTensor.setZero();

    for (auto &tra : traj) {
      ValueBatch advTra = tra.getGAE(vfunction, task->discountFtr(), lambda, task->termValue());
      if (normalize){
        rai::Math::MathFunc::normalize(advTra);
      }
      if(advantageTensor.dim(0) == 1){
        //BatchN = DataN
        advantageTensor.block(0, dataID, 1, advTra.cols()) = advTra;
      } else {
        //RNN
        advantageTensor.block(0, batchID, advTra.cols(),1) = advTra.transpose();
      }
      dataID +=advTra.cols();
      batchID++;
    }
  }

  bool iterateBatch(const int batchSize = 0, bool isrecurrent = false, bool shuffle = false){
    int cur_batch_size = batchSize;
    if (cur_batch_size >=  batchN - cur_ID || cur_batch_size == 0) {
      cur_batch_size =  batchN - cur_ID;
    }
    if(cur_ID >= batchN){
      cur_ID = 0;
      return false;
    }
    if (cur_ID == 0 && shuffle)
      ///this->shuffleBatch;
      /// permute batches,,,
      ;

    cur_minibatch.states.resize(stateTensor.dim(0),stateTensor.dim(1),cur_batch_size);
    cur_minibatch.actions.resize(actionTensor.dim(0),actionTensor.dim(1),cur_batch_size);
    cur_minibatch.actionNoises.resize(actionNoiseTensor.dim(0),actionNoiseTensor.dim(1),cur_batch_size);
    cur_minibatch.advantages.resize(advantageTensor.dim(0),cur_batch_size);
    if(isrecurrent) cur_minibatch.lengths.resize(cur_batch_size);

    cur_minibatch.states = stateTensor.batchBlock(cur_ID, cur_batch_size);
    cur_minibatch.actions = actionTensor.batchBlock(cur_ID,cur_batch_size);
    cur_minibatch.actionNoises = actionNoiseTensor.batchBlock(cur_ID,cur_batch_size);
    cur_minibatch.advantages = advantageTensor.block(0, cur_ID, advantageTensor.dim(0),cur_batch_size);
    if(isrecurrent) cur_minibatch.lengths = trajLength.block(cur_ID,cur_batch_size);

    cur_ID +=cur_batch_size;
//    LOG(INFO) << "cur_ID/batchN = " << cur_ID << "/" << batchN << " batchsize = " << cur_batch_size;

    return true;
  };

  /////////////////////////// Core
  TrajAcquisitor_ *trajAcquisitor_ = nullptr;
  TupleAcquisitor_ *tupleAcquisitor_ = nullptr;
  std::vector<Trajectory_> traj;
  int dataN;
  int cur_ID;
  int batchN;

  /////////////////////////// batches
  StateBatch stateBat, termStateBat;
  ValueBatch valueBat, termValueBat;

  tensorBatch cur_minibatch;

  Tensor<Dtype, 1> trajLength;
  Tensor<Dtype, 1> termType;
  Tensor<Dtype, 2> advantageTensor;
  Tensor<Dtype, 2> costTensor;
  Tensor<Dtype, 2> valueTensor;

  Tensor<Dtype, 3> stateTensor;
  Tensor<Dtype, 3> hiddenStateTensor;
  Tensor<Dtype, 3> actionTensor;
  Tensor<Dtype, 3> actionNoiseTensor;

 private:
  void sampleBatchOfInitial(StateBatch &initial, std::vector<Task_ *> &task) {
    for (int trajID = 0; trajID < initial.cols(); trajID++) {
      State state;
      task[0]->setToInitialState();
      task[0]->getState(state);
      initial.col(trajID) = state;
    }
  }

  void processTrajs(Task_ *task,
                    Policy_ *policy,
                    ValueFunc_ *vfunction = nullptr) {

    dataN = 0;
    int maxlen = 0;

    for (auto &tra : traj) dataN += tra.size() - 1;

    stateBat.resize(StateDim, dataN);

    int colID = 0;
    for (int traID = 0; traID < traj.size(); traID++) {
      for (int timeID = 0; timeID < traj[traID].size() - 1; timeID++) {
        stateBat.col(colID++) = traj[traID].stateTraj[timeID];
      }
    }

    if (policy->isRecurrent()) {
      /////Zero padding tensor//////////////////
      for (auto &tra : traj)
        if (maxlen < tra.stateTraj.size() - 1) maxlen = int(tra.stateTraj.size()) - 1;

      batchN= int(traj.size());
      stateTensor.resize(StateDim, maxlen, batchN);
      actionTensor.resize(ActionDim, maxlen, batchN);
      actionNoiseTensor.resize(ActionDim, maxlen, batchN);
      costTensor.resize(maxlen, batchN);
      trajLength.resize(batchN);
      termType.resize(batchN);

      stateTensor.setZero();
      actionTensor.setZero();
      actionNoiseTensor.setZero();
      costTensor.setZero();

      for (int i = 0; i < batchN; i++) {
        trajLength[i] = traj[i].stateTraj.size() - 1;
        stateTensor.partiallyFillBatch(i, traj[i].stateTraj, 1);
        actionTensor.partiallyFillBatch(i, traj[i].actionTraj, 1);
        actionNoiseTensor.partiallyFillBatch(i, traj[i].actionNoiseTraj, 1);
        for (int timeID = 0; timeID < traj[i].size() - 1; timeID++){
          costTensor.eMat()(timeID,i) = traj[i].costTraj[timeID];
        }
        termType[i] = Dtype(traj[i].termType);
      }


    } else {
      maxlen = 1;
      stateTensor.resize(StateDim, 1, dataN);
      actionTensor.resize(ActionDim, 1, dataN);
      actionNoiseTensor.resize(ActionDim, 1, dataN);
      stateTensor.setZero();
      actionTensor.setZero();
      actionNoiseTensor.setZero();
//      trajLength.resize(1);
//      trajLength.setConstant(dataN);
      batchN = dataN;

      int pos = 0;
      for (int traID = 0; traID < traj.size(); traID++) {
        for (int timeID = 0; timeID < traj[traID].size() - 1; timeID++) {
          actionTensor.batch(pos) = traj[traID].actionTraj[timeID];
          actionNoiseTensor.batch(pos++) = traj[traID].actionNoiseTraj[timeID];
        }
      }
      stateTensor.copyDataFrom(stateBat);
    }
    advantageTensor.resize(maxlen, batchN);

    // update terimnal value
    if (vfunction) {
      termValueBat.resize(1, traj.size());
      termStateBat.resize(StateDim, traj.size());
      valueBat.resize(dataN);
//      valueTensor.resize(maxlen, batchN);

      for (int traID = 0; traID < traj.size(); traID++)
        termStateBat.col(traID) = traj[traID].stateTraj.back();
      vfunction->forward(termStateBat, termValueBat);
      for (int traID = 0; traID < traj.size(); traID++)
        if (traj[traID].termType == TerminationType::timeout) {
        traj[traID].updateValueTrajWithNewTermValue(termValueBat(traID), task->discountFtr());
        }

      colID = 0;
      for (int traID = 0; traID < traj.size(); traID++) {
        for (int timeID = 0; timeID < traj[traID].size() - 1; timeID++){
//          if(maxlen != 1) advantageTensor.eMat()(colID,traID) = traj[traID].valueTraj[timeID];
          valueBat(colID++) = traj[traID].valueTraj[timeID];
        }
      }
//      if(maxlen == 1) valueTensor = valueBat;
    }
  }
};

}
}

#endif //RAI_LearningData_HPP
