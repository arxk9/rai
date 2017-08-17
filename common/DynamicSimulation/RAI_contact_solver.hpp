//
// Created by jhwangbo on 3/12/17.
//

#ifndef RAI_GAUSSSEIDAL5_HPP
#define RAI_GAUSSSEIDAL5_HPP

#include <vector>
#include <stdlib.h>
#include <iostream>
#include <boost/bind.hpp>
#include <math/RAI_math.hpp>
#include <math.h>
#include "Eigen/Core"
#include "UnilateralContact.hpp"
#include "math/inverseUsingCholesky.hpp"
#include "math/RandomNumberGenerator.hpp"
#include "math/GoldenSectionMethod.hpp"
#include "math/RAI_math.hpp"

namespace RAI {
namespace Dynamics {

class RAI_contact_solver {

  typedef Eigen::Vector3d Vector3d;
  typedef Eigen::Vector2d Vector2d;
  typedef Eigen::Matrix3d Matrix3d;
  typedef Eigen::VectorXd VectorXd;

 public:
  RAI_contact_solver() { eye3.setIdentity(); }
  ~RAI_contact_solver() {}

  void solve(std::vector<UnilateralContact *> &uniContacts,
             const MatrixXd &M_inv, const VectorXd &tauStar) {

    unsigned contactN = uniContacts.size();
    unsigned stateDim = M_inv.cols();
    alpha = 0.8;

    std::vector<Vector3d> c(contactN);
    std::vector<Vector3d> oldImpulseUpdate(contactN, Vector3d::Zero());
    std::vector<Vector3d> impulseUpdate(contactN);
    std::vector<Vector3d> newImpulse(contactN);

    inertia.resize(contactN);
    inertiaInv.resize(contactN);

    for (unsigned i = 0; i < contactN; i++) {
      inertiaInv[i].resize(contactN);
      c[i] = uniContacts[i]->jaco * tauStar;
      for (unsigned j = 0; j < contactN; j++)
        inertiaInv[i][j] = uniContacts[i]->jaco * (M_inv * uniContacts[j]->jaco.transpose());
      cholInv(inertiaInv[i][i], inertia[i]);
//      inertia[i] = inertiaInv[i][i].llt().solve(eye3);
    }

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
          new_z_vel = uniContacts[i]->vel(2) * (1.0 - alpha);
          newVel(2) = new_z_vel;
          newVel.head(2).setZero();
          impulseUpdate[i] = -uniContacts[i]->impulse;
          zeroMotionImpulse = inertia[i] * (newVel - rest);
          ftan = zeroMotionImpulse.head(2).norm();

          if (zeroMotionImpulse(2) * uniContacts[i]->mu > ftan) {
            /// if no slip take a step toward the non-slip impulse
            newVel.head(2) = uniContacts[i]->vel.head(2) * (1.0 - alpha);
            uniContacts[i]->impulse = inertia[i] * (newVel - rest);
          } else {
            uniContacts[i]->vel(2) = new_z_vel;
            uniContacts[i]->impulse = inertia[i] * (uniContacts[i]->vel - rest);
            /// if it is going to slip
            ftan = uniContacts[i]->impulse.head(2).norm();
            uniContacts[i]->vel = inertiaInv[i][i] * uniContacts[i]->impulse + rest;
            if (uniContacts[i]->impulse(2) * uniContacts[i]->mu < ftan)
              projectToFeasibleSet2(uniContacts[i], rest, new_z_vel, i, ftan);
            uniContacts[i]->vel = inertiaInv[i][i] * uniContacts[i]->impulse + rest;
            direction.head(2) = uniContacts[i]->vel.head(2);
            direction(2) = (inertiaInv[i][i](2,0) * direction(0) + inertiaInv[i][i](2,1) * direction(1)) / -inertiaInv[i][i](2,2);
            direction /= -direction.norm();

            /// compute the magnitude of the correction impulse that descreases contact energy by alpha
            Ek = uniContacts[i]->vel.transpose() * (inertia[i] * uniContacts[i]->vel);
            quad_a = (direction.transpose() * (inertiaInv[i][i] * direction));
            quad_b = 2.0 * (direction.transpose() * uniContacts[i]->vel)(0);
            quad_c = (1.0 - alpha) * Ek;
            quad_determinant = quad_b * quad_b - 4.0 * quad_a * quad_c;
            correctionMagnitude = ( (quad_determinant > 0) ? (-quad_b - std::sqrt(quad_determinant))/(2.0 * quad_a) : (-quad_b/(2.0 * quad_a) ) );
            impulseBeforeCorrection = uniContacts[i]->impulse;
            sign_0 = signOf(uniContacts[i]->impulse, uniContacts[i]->vel);

            uniContacts[i]->impulse += correctionMagnitude * direction;
            uniContacts[i]->vel = inertiaInv[i][i] * uniContacts[i]->impulse + rest;
            ftan = uniContacts[i]->impulse.head(2).norm();

            /// while there is a slip, perform the line search for angle minimization
            while (uniContacts[i]->impulse(2) * uniContacts[i]->mu < ftan) {
              projectToFeasibleSet2(uniContacts[i], rest, new_z_vel, i, ftan);
              /// take velocity step
              uniContacts[i]->vel = inertiaInv[i][i] * uniContacts[i]->impulse + rest;

              if (signOf(uniContacts[i]->impulse, uniContacts[i]->vel) == sign_0 || correctionMagnitude < 1e-20) {
                correctionMagnitude *= alpha;
                uniContacts[i]->impulse = impulseBeforeCorrection + correctionMagnitude * direction;
                ftan = uniContacts[i]->impulse.head(2).norm();
                projectToFeasibleSet2(uniContacts[i], rest, new_z_vel, i, ftan);
                break;
              }
              correctionMagnitude *= 0.7;
              uniContacts[i]->impulse = impulseBeforeCorrection + correctionMagnitude * direction;
              ftan = uniContacts[i]->impulse.head(2).norm();
            }
          }
          uniContacts[i]->vel = inertiaInv[i][i] * uniContacts[i]->impulse + rest;
          impulseUpdate[i] += uniContacts[i]->impulse;
          velError = std::min(uniContacts[i]->vel(2), 0.0);
          error += impulseUpdate[i].squaredNorm() + velError * velError;
        } else {
          uniContacts[i]->impulse = (1.0-alpha) * uniContacts[i]->impulse;
          error += (uniContacts[i]->impulse / (alpha * (1.0-alpha))).squaredNorm();
        }
      }

      if (error < 1e-10) {
//        std::cout<<counterToBreak<<std::endl;
        break;
      }
      if (++counterToBreak > 1000) {
        if(error > 1e-5) {
          std::cout << "the Contact dynamic solver did not converge in time, the error is " << error << std::endl;
          std::cout << "the Contact force is " << uniContacts[0]->impulse.transpose() << std::endl;
        }
        break;
      }

      alpha = (alpha - alpha_low) * alpha_decay + alpha_low;
    }

  }

 private:

  inline void projectToFeasibleSet(UnilateralContact *contact, Vector3d& rest, double new_z_vel, int contactID, double ftan) {
    ex = contact->impulse(0) / ftan;
    ey = contact->impulse(1) / ftan;
    w0exPw1ey = inertiaInv[contactID][contactID](2,0) * ex + inertiaInv[contactID][contactID](2,1) * ey;
    ftanProj = (new_z_vel - rest(2)) / (w0exPw1ey + inertiaInv[contactID][contactID](2,2) / contact->mu);
    contact->impulse(0) = ex * ftanProj;
    contact->impulse(1) = ey * ftanProj;
    contact->impulse(2) = ftanProj / contact->mu;
  }

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

  Eigen::Matrix3d eye3;
  std::vector<std::vector<Matrix3d> > inertiaInv;
  std::vector<Matrix3d> inertia;
  Vector3d rest, newVel;
  double new_z_vel;
  double alpha;
  double alpha_low = 0.1;
  double alpha_decay = 0.99;
  double ex;
  double ey;
  double w0exPw1ey;
  double ftanProj;
  Vector3d zeroMotionImpulse;
  double velError;
  double ftan;

  /// slippage related
  Vector3d direction;
  Vector3d impulseCorrection;
  Vector3d impulseBeforeCorrection;
  double Ek;
  double quad_a;
  double quad_b;
  double quad_c;
  double quad_determinant;
  double correctionMagnitude;
  double fvMisalignment_0;
  bool sign_0;

};

}
}

#endif //RAI_GAUSSSEIDAL_HPP
