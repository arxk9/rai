//
// Created by jhwangbo on 3/23/17.
//

#ifndef RAI_TRAJECTORYACQUISITOR_HPP
#define RAI_TRAJECTORYACQUISITOR_HPP
#include "Acquisitor.hpp"
#include "rai/memory/Trajectory.hpp"
#include "rai/noiseModel/Noise.hpp"
#include "rai/tasks/common/Task.hpp"
#include "rai/function/common/Policy.hpp"
#include "rai/memory/ReplayMemorySARS.hpp"

namespace RAI {
namespace ExpAcq {

template<typename Dtype, int StateDim, int ActionDim>
class TrajectoryAcquisitor : public Acquisitor<Dtype, StateDim, ActionDim> {

  using Task_ = Task::Task<Dtype, StateDim, ActionDim, 0>;
  using Trajectory = Memory::Trajectory<Dtype, StateDim, ActionDim>;
  using Noise_ = Noise::Noise<Dtype, ActionDim>;
  using Policy_ = FuncApprox::Policy<Dtype, StateDim, ActionDim>;
  using StateBatch = Eigen::Matrix<Dtype, StateDim, -1>;
  using ReplayMemory_ = Memory::ReplayMemorySARS<Dtype, StateDim, ActionDim>;

 public:
  virtual Dtype acquire(std::vector<Task_ *> &taskset,
                        Policy_ *policy,
                        std::vector<Noise_ *> &noise,
                        std::vector<Trajectory> &trajectorySet,
                        StateBatch &startingState,
                        double timeLimit,
                        bool countStep,
                        ReplayMemory_ *memory = nullptr) = 0;

};

}
}

#endif //RAI_TRAJECTORYACQUISITOR_HPP
