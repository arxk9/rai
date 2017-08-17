//
// Created by jhwangbo on 22.09.16.
//

#include <rai/RAI_core>
#include "rai/tasks/poleBalancing/PoleBalancing.hpp"
#include "rai/noiseModel/OrnsteinUhlenbeckNoise.hpp"
#include "rai/function/tensorflow/Qfunction_TensorFlow.hpp"
#include "rai/function/tensorflow/DeterministicPolicy_TensorFlow.hpp"
#include "rai/algorithm/DDPG.hpp"
#include <rai/experienceAcquisitor/ExperienceTupleAcquisitor_Sequential.hpp>

/// learning states
using Dtype = float;

/// shortcuts
using RAI::Task::ActionDim;
using RAI::Task::StateDim;

using Task = RAI::Task::PoleBalancing<Dtype>;
using Policy_TensorFlow = RAI::FuncApprox::DeterministicPolicy_TensorFlow<Dtype, StateDim, ActionDim>;
using Qfunction_TensorFlow = RAI::FuncApprox::Qfunction_TensorFlow<Dtype, StateDim, ActionDim>;
using ReplayMemorySARS = RAI::Memory::ReplayMemorySARS<Dtype, StateDim, ActionDim>;
using Acquisitor_ = RAI::ExpAcq::ExperienceTupleAcquisitor_Sequential<Dtype, StateDim, ActionDim>;
using Noise = RAI::Noise::Noise<Dtype, ActionDim>;
using OUNoise = RAI::Noise::OrnsteinUhlenbeck<Dtype, ActionDim>;
#define nThread 2

int main(int argc, char *argv[]) {

  RAI_init();
  omp_set_num_threads(nThread);

  Task task;
  OUNoise noise(0.15, 0.3);
  ReplayMemorySARS replayMemorySARS(1000000);
  Acquisitor_ acquisitor;

  ////////////////////////// Define Function approximations //////////
  Policy_TensorFlow policy("cpu", "MLP", "relu 1e-3 3 32 32 1", 1e-4);
  Policy_TensorFlow policy_target("cpu", "MLP", "relu 1e-3 3 32 32 1", 1e-4);
  Qfunction_TensorFlow qfunction("cpu", "MLP2", "relu 1e-3 3 1 32 32 1", 1e-3);
  Qfunction_TensorFlow qfunction_target("cpu", "MLP2", "relu 1e-3 3 1 32 32 1", 1e-3);

  ////////////////////////// Algorithm ////////////////////////////////
  std::vector<RAI::Task::Task<Dtype,StateDim,ActionDim,0> *> taskVector = {&task};
  std::vector<Noise*> noiseVector = {&noise};
  RAI::Algorithm::DDPG<Dtype, StateDim, ActionDim>
      algorithm(taskVector, &qfunction, &qfunction_target, &policy, &policy_target, noiseVector, &acquisitor, &replayMemorySARS, 80, 1, 1e-3);
  algorithm.setVisualizationLevel(1);

  ////////////////////////// Learning /////////////////////////////////
  algorithm.initiallyFillTheMemory();
  for (int iterationNumber = 0; iterationNumber < 10; iterationNumber++)
    algorithm.learnForNSteps(5000);
}