//
// Created by jhwangbo on 03.03.17.
//
#include "TypeDef.hpp"

#ifndef RAI_CONTACT_HPP
#define RAI_CONTACT_HPP

namespace RAI {
namespace Dynamics {

class Contact {

 public:

  Contact() {
    impulse.setZero();
  }

  MatrixXd jaco;
  Eigen::Vector3d normal;
  Eigen::Vector3d impulse;
  Eigen::Vector3d vel;

  //// hold optimization state for some methods
  Eigen::Vector3d x;
  Eigen::Matrix3d dfdx;

};

}
}
#endif //RAI_CONTACT_HPP
