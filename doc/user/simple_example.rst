========================
Simple Example
========================

Examples can be found in :code:`applications/examples`. The following example code learns a policy for poleBalancing task with DDPG *(Lilicrap et. al. 2016)*::

  #include "rai/RAI_core"
  #include "rai/tasks/poleBalancing/PoleBalancing.hpp"
  #include "rai/noiseModel/OrnsteinUhlenbeckNoise.hpp"
  #include "rai/function/tensorflow/Qfunction_TensorFlow.hpp"
  #include "rai/function/tensorflow/DeterministicPolicy_TensorFlow.hpp"
  #include "rai/algorithm/DDPG.hpp"
  #include "rai/experienceAcquisitor/ExperienceTupleAcquisitor_Sequential.hpp"


  /// learning states
  using Dtype = float;

  /// shortcuts
  using RAI::Task::ActionDim;
  using RAI::Task::StateDim;
  using RAI::Task::CommandDim;

  using Task = RAI::Task::PoleBalancing<Dtype>;
  using Policy_TensorFlow = RAI::FuncApprox::DeterministicPolicy_TensorFlow<Dtype, StateDim, ActionDim>;
  using Qfunction_TensorFlow = RAI::FuncApprox::Qfunction_TensorFlow<Dtype, StateDim, ActionDim>;
  using ReplayMemorySARS = RAI::Memory::ReplayMemorySARS<Dtype, StateDim, ActionDim>;
  using Acquisitor_ = RAI::ExpAcq::ExperienceTupleAcquisitor_Sequential<Dtype, StateDim, ActionDim>;
  using Noise = RAI::Noise::Noise<Dtype, ActionDim>;

  int main() {

    RAI_init();

    Task task(Task::fixed, Task::easy);
    RAI::Noise::OrnsteinUhlenbeck<Dtype, ActionDim> noise(0.15, 0.3);
    RAI::Memory::ReplayMemorySARS<Dtype, StateDim, ActionDim> replayMemorySARS(1000000);

    Policy_TensorFlow policy("cpu", "MLP", "relu 1e-3 3 32 32 1", 1e-4);
    Policy_TensorFlow policy_target("cpu", "MLP", "relu 1e-3 3 32 32 1", 1e-4);
    Qfunction_TensorFlow qfunction("cpu", "MLP2", "relu 1e-3 3 1 32 32 1", 1e-3);
    Qfunction_TensorFlow qfunction_target("cpu", "MLP2", "relu 1e-3 3 1 32 32 1", 1e-3);

    Acquisitor_ acquisitor;

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

The code is fairly simple. We create the core elements which are,

1. Task
2. Noise
3. Memory
4. Functions
5. Experience Acquisitor
6. Algorithm

The pointers from 1~5 are given to the algorithm constructor so that the algorithm can use them internally.
We set the visualization level to 1 to check the performance after each iteration (setting it to 0 skips visual feedback).
Then we run DDPG methods for learning.
Note that memory might not be necessary for some algorithms that instantiate their own memory objects.
:code:`RAI_init()` must be called in the first line of the main function.
It generates logging directory and sets global variables that are used in other files.
