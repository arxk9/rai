//
// Created by jhwangbo on 3/12/17.
//

#ifndef RAI_RAICONTACT2_HPP
#define RAI_RAICONTACT2_HPP

#include <vector>
#include <stdlib.h>
#include <iostream>
#include <boost/bind.hpp>
#include <rai/common/math/RAI_math.hpp>
#include <math.h>
#include "Eigen/Core"
#include "UnilateralContact.hpp"
#include "rai/common/math/inverseUsingCholesky.hpp"
#include "rai/RAI_core"

namespace rai {
namespace Dynamics {

class RAI_contact_solver2 {

  typedef Eigen::Vector3d Vector3d;
  typedef Eigen::Vector2d Vector2d;
  typedef Eigen::Matrix3d Matrix3d;
  typedef Eigen::Matrix<double, 3, 2> Mat32d;
  typedef Eigen::VectorXd VectorXd;
  typedef Eigen::RowVector2d RowVec2d;
  typedef Eigen::Vector2d Vec2d;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  RAI_contact_solver2() {}

  ~RAI_contact_solver2() {}

  void solve(std::vector<UnilateralContact *> &uniContacts,
             const MatrixXd &M_inv, const VectorXd &tauStar) {

    uniContacts_ = &uniContacts;

    unsigned contactN = uniContacts.size();
    unsigned stateDim = M_inv.cols();
    alpha = 1.0;
    oneMinusAlpha = 1.0-alpha;

    std::vector<Vector3d> c(contactN);
    std::vector<Vector3d> normVec(contactN);
    std::vector<std::string> slipCondtion(contactN);
    double negativeMuSquared[contactN];

    inertia.resize(contactN);
    inertiaInv.resize(contactN);
    inertiaInv_red.resize(contactN);
    n2_mu.resize(contactN);

    MatrixXd temp(3, stateDim);
    RowVec2d normVecRatio;

    for (unsigned i = 0; i < contactN; i++) {
      inertiaInv[i].resize(contactN);
      c[i] = uniContacts[i]->jaco * tauStar;
      temp = uniContacts[i]->jaco * M_inv;
      for (unsigned j = 0; j < contactN; j++)
        inertiaInv[i][j] = temp * uniContacts[j]->jaco.transpose();
      normVec[i] = inertiaInv[i][i].row(2).transpose();
      negativeMuSquared[i] = -uniContacts[i]->mu * uniContacts[i]->mu;
      normVecRatio << inertiaInv[i][i](2,0)/inertiaInv[i][i](2,2), inertiaInv[i][i](2,1)/inertiaInv[i][i](2,2);
      inertiaInv_red[i] = inertiaInv[i][i].leftCols(2) - inertiaInv[i][i].col(2) * normVecRatio;
      n2_mu[i] = inertiaInv[i][i](2,2)/uniContacts[i]->mu;
    }

    for (unsigned i = 0; i < contactN; i++)
      cholInv(inertiaInv[i][i], inertia[i]);

    int counterToBreak = 0;
    double error = 1.0;

    while (error > 1e-10 && ++counterToBreak < 1000) {
      error = 0.0;
      for (unsigned i = 0; i < contactN; i++) {

        /// initially compute velcoties
        rest = c[i];
        for (unsigned j = 0; j < contactN; j++)
          if (i != j) rest += inertiaInv[i][j] * uniContacts[j]->impulse;

        /// if it is on ground
        if (rest(2) < 0) {
          originalImpulse = uniContacts[i]->impulse; /// storing previous impulse for impulse update computation.
          /// This is only to check the terminal condition
          uniContacts[i]->impulse = inertia[i] * -rest;

          if (uniContacts[i]->impulse(2) * uniContacts[i]->mu < uniContacts[i]->impulse.head(2).norm()) {
            double theta = std::atan2(uniContacts[i]->impulse(1), uniContacts[i]->impulse(0));
            normToCone << uniContacts[i]->impulse(0), uniContacts[i]->impulse(1), negativeMuSquared[i]
              * uniContacts[i]->impulse(2);

            /// if it is going to slip
            newdirection = normVec[i].cross(normToCone);
            double cTheta = cos(theta), sTheta = sin(theta);
            double r =  -rest(2)/(n2_mu[i] +inertiaInv[i][i](2,0)*cTheta +inertiaInv[i][i](2,1)*sTheta);
            Vec2d rscTheta(r * cTheta, r * sTheta);
            newRest = rest - inertiaInv[i][i].col(2) * (rest(2)/inertiaInv[i][i](2,2));
            newgradient1D = newdirection.dot(inertiaInv_red[i]*rscTheta + newRest);

            double oldTheta, midTheta;
            /// fixed angle stepping
            angleStep = -0.005 * sgn(newgradient1D);
            do {
              gradient1D = newgradient1D;
              oldTheta = theta;
              theta += angleStep;
              cTheta = cos(theta);
              sTheta = sin(theta);
              r = -rest(2)/(n2_mu[i] +inertiaInv[i][i](2,0)*cTheta +inertiaInv[i][i](2,1)*sTheta);
              normToCone << r*cTheta, r*sTheta, r*uniContacts[i]->negMu;
              newdirection = normVec[i].cross(normToCone);
              newgradient1D = newdirection.dot(inertiaInv_red[i]*normToCone.head(2) + newRest);
              angleStep *= 2.0;
            } while (gradient1D * newgradient1D > 0.0);
            /// bisection method
            do {
              rscTheta = normToCone.head(2);
              midTheta = (oldTheta + theta) / 2.0;
              cTheta = cos(midTheta);
              sTheta = sin(midTheta);
              r = -rest(2)/(n2_mu[i] +inertiaInv[i][i](2,0)*cTheta +inertiaInv[i][i](2,1)*sTheta);
              normToCone << r*cTheta, r*sTheta, r*uniContacts[i]->negMu;
              newdirection = normVec[i].cross(normToCone);
              newgradient1D = newdirection.dot(inertiaInv_red[i]*normToCone.head(2) + newRest);

              if (gradient1D * newgradient1D > 0)
                oldTheta = midTheta;
              else
                theta = midTheta;
            } while ((rscTheta - normToCone.head(2)).squaredNorm() > 1e-11);

            uniContacts[i]->impulse<<normToCone.head(2), r/uniContacts[i]->mu;

          }
          uniContacts[i]->impulse = originalImpulse * oneMinusAlpha + uniContacts[i]->impulse * alpha;
          velError = std::min(normVec[i].dot(uniContacts[i]->impulse) + rest(2), 0.0);
          originalImpulse -= uniContacts[i]->impulse;
          error += originalImpulse.squaredNorm() + velError * velError;
        } else {
          uniContacts[i]->impulse *= oneMinusAlpha;
          error += uniContacts[i]->impulse.squaredNorm();
        }
      }

      alpha = (alpha - alpha_low) * alpha_decay + alpha_low;
      oneMinusAlpha = 1-alpha;
    }
  }

 private:

  template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
  }

  std::vector<std::vector<Matrix3d> > inertiaInv;
  std::vector<Mat32d> inertiaInv_red;
  std::vector<double> n2_mu;

  std::vector<Matrix3d> inertia;
  Vector3d rest, originalImpulse, newRest;
  Vec2d oldImpulseTan;
  double alpha, oneMinusAlpha;
  double alpha_low = 0.5;
  double alpha_decay = 0.99;
  Vector3d normToCone;
  double velError;

  /// slippage related
  Vector3d direction, gradient, newdirection;
  double gradient1D, newgradient1D, angleStep;
  Vector3d oldImpulse, nextImpulse;
  std::vector<UnilateralContact *> *uniContacts_;

};

}
}

#endif //RAICONTACT2
