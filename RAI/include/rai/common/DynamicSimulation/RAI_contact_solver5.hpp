//
// Created by jhwangbo on 21/07/17.
//

#ifndef RAI_RAI_CONTACT_SOLVER5_HPP
#define RAI_RAI_CONTACT_SOLVER5_HPP

#include <vector>
#include <stdlib.h>
#include <iostream>
#include <boost/bind.hpp>
#include <rai/common/math/RAI_math.hpp>
#include <math.h>
#include "Eigen/Core"
#include "UnilateralContact.hpp"
#include "rai/common/math/inverseUsingCholesky.hpp"
#include "rai/common/math/RandomNumberGenerator.hpp"
#include "rai/common/math/GoldenSectionMethod.hpp"
#include "rai/common/math/RAI_math.hpp"

namespace RAI {
namespace Dynamics {

class RAI_contact_solver5 {

  typedef Eigen::Vector3d Vector3d;
  typedef Eigen::Vector2d Vector2d;
  typedef Eigen::Matrix3d Matrix3d;
  typedef Eigen::VectorXd VectorXd;

 public:
  RAI_contact_solver5() {
//    Utils::logger->addVariableToLog(1, "ctb", "counter-To-break of cs5");
  }
  ~RAI_contact_solver5() {}

  void solve(std::vector<UnilateralContact *> &uniContacts,
             const MatrixXd &M_inv, const VectorXd &tauStar) {

    unsigned contactN = uniContacts.size();
    unsigned stateDim = M_inv.cols();
    alpha = 0.9;
    oneMinusAlpha = 1-alpha;

    std::vector<Vector3d> c(contactN);
    std::vector<Vector3d> oldImpulseUpdate(contactN, Vector3d::Zero());
    std::vector<Vector3d> impulseUpdate(contactN);
    std::vector<Vector3d> newImpulse(contactN);

    inertia.resize(contactN);
    inertiaInv.resize(contactN);
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

    for (unsigned i = 0; i < contactN; i++)
      cholInv(inertiaInv[i][i], inertia[i]);

    int counterToBreak = 0;
    double error;

    while (true) {
      error = 0.0;

      for (unsigned i = 0; i < contactN; i++) {
        /// initially compute velcoties
        rest = c[i];
        for (unsigned j = 0; j < contactN; j++)
          if (i!=j) rest += inertiaInv[i][j] * uniContacts[j]->impulse;

        /// if it is on ground
        if (rest(2) < 0) {
          uniContacts[i]->vel = inertiaInv[i][i] * uniContacts[i]->impulse + rest;
          new_z_vel = uniContacts[i]->vel(2) * oneMinusAlpha;
          newVel(2) = new_z_vel;
          newVel.head(2).setZero();
          impulseUpdate[i] = -uniContacts[i]->impulse;
          uniContacts[i]->impulse = inertia[i] * (newVel - rest);
          ftan = uniContacts[i]->impulse.head(2).norm();
          if (uniContacts[i]->impulse(2) * uniContacts[i]->mu > ftan) {
            /// if no slip take a step toward the non-slip impulse
            newVel.head(2) = uniContacts[i]->vel.head(2) * oneMinusAlpha;
            uniContacts[i]->impulse = inertia[i] * (newVel - rest);
          } else {
            uniContacts[i]->vel(2) = new_z_vel;
            uniContacts[i]->impulse = inertia[i] * (uniContacts[i]->vel - rest);
            /// if it is going to slip
            ftan = uniContacts[i]->impulse.head(2).norm();
            if (uniContacts[i]->impulse(2) * uniContacts[i]->mu < ftan)
              projectToFeasibleSet2(uniContacts[i], rest, new_z_vel, i, ftan);
            uniContacts[i]->vel = inertiaInv[i][i] * uniContacts[i]->impulse + rest;
            normToCone.head(2) = uniContacts[i]->impulse.head(2);
            normToCone(2) = (-uniContacts[i]->mu * uniContacts[i]->mu) * uniContacts[i]->impulse(2);
            eta = inertiaInv[i][i].row(2).transpose().cross(normToCone);
            /// change direction if eta points the same direction as velocity
            eta = ((uniContacts[i]->impulse(0)*eta(1) - uniContacts[i]->impulse(1)*eta(0)) * (uniContacts[i]->impulse(0)*uniContacts[i]->vel(1) - uniContacts[i]->impulse(1)*uniContacts[i]->vel(0)) > 0.0 ? -1 : 1) * eta;
            eta /= eta.norm();
            vel_direction = inertiaInv[i][i] * eta;

            const double current_angle = std::acos(uniContacts[i]->vel.head(2).dot(uniContacts[i]->impulse.head(2)) / uniContacts[i]->vel.head(2).norm() / uniContacts[i]->impulse.head(2).norm());
//            std::cout << "current_angle " << M_PI - current_angle << std::endl;
            /// update only if the the vel and impulse are not anti-aligned
            if(M_PI - current_angle > 1e-7) {
              const double targetAngle = (M_PI - current_angle) * alpha * 0.5;
//              std::cout << "targetAngle " << targetAngle << std::endl;

              /// for debugging
//              double angle_imp_impP = M_PI - std::acos(
//                  uniContacts[i]->impulse.head(2).dot(eta.head(2)) / uniContacts[i]->impulse.head(2).norm()
//                      / eta.head(2).norm());
//              double beta_1 = uniContacts[i]->impulse.head(2).norm() / std::sin(M_PI - targetAngle - angle_imp_impP)
//                  * std::sin(targetAngle);
//              double angle_v_vp = M_PI - std::acos(
//                  uniContacts[i]->vel.head(2).dot(vel_direction.head(2)) / uniContacts[i]->vel.head(2).norm()
//                      / vel_direction.head(2).norm());
//              double beta_2 = uniContacts[i]->vel.head(2).norm() / std::sin(M_PI - targetAngle - angle_v_vp)
//                  * std::sin(targetAngle) / vel_direction.norm();
//              uniContacts[i]->impulse += std::min(beta_1, beta_2) * eta;

              const double singTargetAngle = std::sin(targetAngle);
              const double impulsehead2norm = uniContacts[i]->impulse.head(2).norm();
              const double velhead2norm = uniContacts[i]->vel.head(2).norm();
              /// optimized
              uniContacts[i]->impulse += std::min(impulsehead2norm / std::sin(std::acos(
                  uniContacts[i]->impulse.head(2).dot(eta.head(2)) / impulsehead2norm
                      / eta.head(2).norm()) - targetAngle) * singTargetAngle,
                                                  velhead2norm / std::sin(std::acos(
                                                      uniContacts[i]->vel.head(2).dot(vel_direction.head(2)) / velhead2norm
                                                          / vel_direction.head(2).norm()) - targetAngle) * singTargetAngle / vel_direction.norm()) * eta;
              projectToFeasibleSet2(uniContacts[i], rest, new_z_vel, i, uniContacts[i]->impulse.head(2).norm());
            }
          }
          uniContacts[i]->vel = inertiaInv[i][i] * uniContacts[i]->impulse + rest;
          impulseUpdate[i] += uniContacts[i]->impulse;
          velError = std::min(uniContacts[i]->vel(2), 0.0);
          error += impulseUpdate[i].squaredNorm() + velError * velError;
        } else {
          uniContacts[i]->impulse *= oneMinusAlpha;
          error += uniContacts[i]->impulse.squaredNorm();
        }
      }

      if (error < 1e-12) {
//        Utils::logger->appendData("ctb", counterToBreak);
//        std::cout<<counterToBreak<<std::endl;
        break;
      }
      if (++counterToBreak > 1000) {
//        if(error > 1e-5) {
//          std::cout << "the Contact dynamic solver did not converge in time, the error is " << error << std::endl;
//          std::cout << "the Contact force is " << uniContacts[0]->impulse.transpose() << std::endl;
//        }
        break;
      }

      alpha = (alpha - alpha_low) * alpha_decay + alpha_low;
      oneMinusAlpha = 1-alpha;
    }
//    Utils::timer->stopTimer("Solver");

  }

 private:

  inline void projectToFeasibleSet2(UnilateralContact *contact,
                                    Vector3d &rest,
                                    double new_z_vel,
                                    int contactID,
                                    double ftan) {
    double tanTheta = -((inertiaInv[contactID][contactID](2, 0) * contact->impulse(0)
        + inertiaInv[contactID][contactID](2, 1) * contact->impulse(1)) / ftan)
        / inertiaInv[contactID][contactID](2, 2);
    double f0 = contact->impulse(2) - tanTheta * ftan;
    double newFtan = f0 / (1/contact->mu - tanTheta);
    contact->impulse(2) = f0 + tanTheta * newFtan;
    contact->impulse.head(2) *= newFtan / ftan;
  }

  inline bool signOf(const Vector3d &v1, const Vector3d &v2) {
    return signbit(v1(0)*v2(1)) - (v1(1)*v2(0));
  }

  std::vector<std::vector<Matrix3d> > inertiaInv;
  std::vector<Matrix3d> inertia;
  Vector3d normToCone, eta, vel_direction;
  Vector3d rest, newVel;
  double new_z_vel;
  double alpha, oneMinusAlpha;
  double alpha_low = 0.1;
  double alpha_decay = 0.99;
  double velError;
  double ftan;

};

}
}
#endif //RAI_RAI_CONTACT_SOLVER4_HPP
