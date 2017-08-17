//
// Created by jhwangbo on 10/08/17.
//

#ifndef RAI_DETERMINISTICMODEL_HPP
#define RAI_DETERMINISTICMODEL_HPP

#include "Model.hpp"

namespace RAI {
namespace FuncApprox {

template<typename Dtype, int stateDim, int actionDim>
class DeterministicModel : public virtual Model<Dtype, stateDim, actionDim> {
 public:

  typedef Eigen::Matrix<Dtype, stateDim, 1> State;
  typedef Eigen::Matrix<Dtype, stateDim, -1> StateBatch;
  typedef Eigen::Matrix<Dtype, actionDim, 1> Action;
  typedef Eigen::Matrix<Dtype, actionDim, -1> ActionBatch;

};

}
}

#endif //RAI_DETERMINISTICMODEL_HPP
