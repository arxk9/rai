//
// Created by jhwangbo on 3/12/17.
//

#ifndef RAI_GAUSSSEIDAL_HPP
#define RAI_GAUSSSEIDAL_HPP

#include <vector>
#include <stdlib.h>
#include <iostream>
#include <boost/bind.hpp>
#include <rai/common/math/RAI_math.hpp>
#include "Eigen/Core"
#include "UnilateralContact.hpp"
#include "rai/common/math/inverseUsingCholesky.hpp"
#include "rai/common/math/RandomNumberGenerator.hpp"
#include "rai/common/math/GoldenSectionMethod.hpp"

namespace rai {
namespace Dynamics {

class SOR {

  typedef Eigen::Vector3d Vector3d;
  typedef Eigen::Matrix3d Matrix3d;
  typedef Eigen::VectorXd VectorXd;

 public:
  SOR() {
    eye3.setIdentity();
//    Utils::logger->addVariableToLog(1, "ctb", "counter-To-break of cs2");
  }

  ~SOR() {}

  void solve(rai::Vector<UnilateralContact *> &uniContacts,
             const MatrixXd &M_inv, const VectorXd &tauStar) {

    unsigned contact3 = 3 * uniContacts.size();
    unsigned contactN = uniContacts.size();
    unsigned stateDim = M_inv.cols();
    double alpha = 0.5;

    rai::Vector<Vector3d> c(contactN);
    rai::Vector<rai::Vector<Matrix3d> > inertiaInv(contactN);
    rai::Vector<Vector3d> r(contactN);
    VectorXd progress(contact3);
    Vector3d rest;
    MatrixXd temp(3, stateDim);

//    Utils::timer->startTimer("Delassus");
    for (unsigned i = 0; i < contactN; i++) {
      inertiaInv[i].resize(contactN);
      c[i] = uniContacts[i]->jaco * tauStar;
      temp = uniContacts[i]->jaco * M_inv;
      for (unsigned j = 0; j < contactN; j++)
        inertiaInv[i][j] = temp * uniContacts[j]->jaco.transpose();
    }

//    Utils::timer->stopTimer("Delassus");

//    Utils::timer->startTimer("Solver");
    for (unsigned i = 0; i < contactN; i++){
      r[i](2) = alpha / inertiaInv[i][i](2, 2);
      r[i](0) =
          alpha / ((inertiaInv[i][i](0, 0) > inertiaInv[i][i](1, 1)) ? inertiaInv[i][i](0, 0) : inertiaInv[i][i](1, 1));
      r[i](1) = r[i](0);
    }

      /// Gauss-Seidal
    int counterToBreak = 0;
    rai::Vector<Vector3d> correctionImpulse;
    correctionImpulse.resize(contactN);
    double error = 0.0;
    while (true) {
      error = 0.0;
      for (unsigned i = 0; i < contactN; i++) {

        /// initially compute velcoties
        rest = c[i];
        for (unsigned j = 0; j < contactN; j++)
          if (i != j) rest += inertiaInv[i][j] * uniContacts[j]->impulse;

        /// only if it is on ground
        uniContacts[i]->vel = inertiaInv[i][i] * uniContacts[i]->impulse + rest;
        correctionImpulse[i] = uniContacts[i]->impulse;
        uniContacts[i]->impulse -= r[i].cwiseProduct(uniContacts[i]->vel);
        projectToFeasibleSet(uniContacts[i]);
        correctionImpulse[i] -= uniContacts[i]->impulse;
        uniContacts[i]->vel = inertiaInv[i][i] * uniContacts[i]->impulse + rest;
        double velError = std::min(uniContacts[i]->vel(2), 0.0);
        error += correctionImpulse[i].squaredNorm() + velError * velError;

//        if(uniContacts[i]->impulse(2) > 0 && uniContacts[i]->vel.head(2).norm()>1e-4) {
//          double errorAngle = M_PI - std::acos(uniContacts[i]->vel.head(2).dot(uniContacts[i]->impulse.head(2)) / uniContacts[i]->vel.head(2).norm() / uniContacts[i]->impulse.head(2).norm());
//          error += errorAngle*errorAngle;
//        }
      }
      if (error < 1e-12) {
//        Utils::logger->appendData("ctb", counterToBreak);
        break;
      }

      if (counterToBreak++ > 10000) {
        std::cout << "error is " << error << std::endl;
      }
    }
//    Utils::timer->stopTimer("Solver");

//    std::cout<<"counterToBreak "<<counterToBreak<<std::endl;
  }

 private:

  inline void projectToFeasibleSet(UnilateralContact *contact) {
    if (contact->impulse(2) < 0)
      contact->impulse.setZero();
    else {
      double tanMag = contact->impulse.head(2).norm();
      if (tanMag > contact->impulse(2) * contact->mu)
        contact->impulse.head(2) *= contact->impulse(2) * contact->mu / tanMag;
    }
  }

  Eigen::Matrix3d eye3;

};

}
}

#endif //RAI_GAUSSSEIDAL_HPP
