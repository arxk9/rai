//
// Created by jhwangbo on 3/12/17.
//

#ifndef RAI_RAICONTACT3_HPP
#define RAI_RAICONTACT3_HPP

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
#include "rai/RAI_core"

namespace RAI {
namespace Dynamics {

class RAI_contact_solver2 {

  typedef Eigen::Vector3d Vector3d;
  typedef Eigen::Vector2d Vector2d;
  typedef Eigen::Matrix3d Matrix3d;
  typedef Eigen::VectorXd VectorXd;
  typedef Eigen::Matrix2d Matrix2d;
 public:
  RAI_contact_solver2() {
    eye3.setIdentity();
    fcn_ = std::bind(&RAI_contact_solver2::contactEnergyFcn, this, std::placeholders::_1);

    /////////////////////// Plotting properties ////////////////////////
    energyFP.title = "energy vs angle";
    energyFP.xlabel = "angles";
    energyFP.ylabel = "energy";

    energy_plot.resize(360);
    angles_plot.resize(360);

    for(int dataID = 0; dataID < 360; dataID++)
      angles_plot(dataID) = dataID / 180.0 * M_PI - M_PI;

  }

  ~RAI_contact_solver2() {}

  void solve(std::vector<UnilateralContact *> &uniContacts,
             const MatrixXd &M_inv, const VectorXd &tauStar) {

    Utils::timer->startTimer("matrix initialization");
    uniContacts_ = &uniContacts;

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
    xyInertia.resize(contactN);
    topInertiaInv.resize(contactN);
    unitNormal.resize(contactN);
    majorAxis.resize(contactN);
    minorAxis.resize(contactN);
    slope2dMajor.resize(contactN);
    slope2dMinor.resize(contactN);

    MatrixXd temp(3, stateDim);

    for (unsigned i = 0; i < contactN; i++) {
      inertiaInv[i].resize(contactN);
      c[i] = uniContacts[i]->jaco * tauStar;
      Utils::timer->startTimer("apparent inertia");
      temp = uniContacts[i]->jaco * M_inv;
      for (unsigned j = 0; j < contactN; j++)
        inertiaInv[i][j] = temp * uniContacts[j]->jaco.transpose();
      Utils::timer->stopTimer("apparent inertia");

      Utils::timer->startTimer("matrix inversion");
      cholInv(inertiaInv[i][i], inertia[i]);
      Utils::timer->stopTimer("matrix inversion");

      xyInertia[i] = inertia[i].topLeftCorner(2,2);
      topInertiaInv[i] = inertiaInv[i][i].topRows(2);

      /// eliptical parameters
      unitNormal[i] = inertiaInv[i][i].row(2) / inertiaInv[i][i].row(2).norm();
      double tempSquaredNorm = unitNormal[i](0) * unitNormal[i](0) + unitNormal[i](1) * unitNormal[i](1);
      majorAxis[i] << (1.0-tempSquaredNorm) * unitNormal[i](0), (1.0-tempSquaredNorm) * unitNormal[i](1), -tempSquaredNorm * unitNormal[i](2);
      majorAxis[i] /= majorAxis[i].norm();
      minorAxis[i] << majorAxis[i](1), -majorAxis[i](0), 0;
      minorAxis[i] /= minorAxis[i].norm();
      slope2dMajor[i] = majorAxis[i](2) / (majorAxis[i](0) * majorAxis[i](0) + majorAxis[i](1) * majorAxis[i](1));
      slope2dMinor[i] = minorAxis[i](2) / (minorAxis[i](0) * minorAxis[i](0) + minorAxis[i](1) * minorAxis[i](1));
    }
    Utils::timer->stopTimer("matrix initialization");

    int counterToBreak = 0;
    double error;

    while (true) {
      error = 0.0;
      bool slipping = false;

      for (unsigned i = 0; i < contactN; i++) {

        Utils::timer->startTimer("rest comp");
        /// initially compute velcoties
        rest = c[i];
        for (unsigned j = 0; j < contactN; j++)
          if (i != j) rest += inertiaInv[i][j] * uniContacts[j]->impulse;
        Utils::timer->stopTimer("rest comp");

        /// if it is on ground
        if (rest(2) < 0) {
          impulseUpdate[i] = -uniContacts[i]->impulse; /// storing previous impulse for impulse update computation.
          /// This is only to check the terminal condition
          originalImpulse = uniContacts[i]->impulse;
          uniContacts[i]->impulse = inertia[i] * -rest;
          ftan = uniContacts[i]->impulse.head(2).norm();

          if (uniContacts[i]->impulse(2) * uniContacts[i]->mu > ftan) {
            /// if no slip take a step toward the non-slip impulse
          } else {
            slipping = true;

            /// triangles bisecting the cone along the major axis
            double centerHeight = -rest(2) / inertiaInv[i][i](2,2);
            double majorRadiusBottomLength1 = centerHeight / (1.0 / uniContacts[i]->mu + slope2dMajor[i]);
            double majorRadiusBottomLengthSum = 2.0 * centerHeight / uniContacts[i]->mu;
            double majorRadiusBottomOffset = majorRadiusBottomLength1 - majorRadiusBottomLengthSum / 2.0;
            double majorCsc = majorAxis[i](0) * majorAxis[i](0) + majorAxis[i](1) * majorAxis[i](1);
            double majorRadiusSlantSide = majorRadiusBottomOffset / majorCsc;
            double majorRadiusHeightFromCenter = majorRadiusBottomOffset / majorCsc * majorAxis[i](2);
            double majorRadius = centerHeight / uniContacts[i]->mu / majorCsc;
            Vector3d center;
            center = majorRadiusSlantSide * majorAxis[i];

            /// minor triangle
            double minorRadiusHeight = majorRadiusHeightFromCenter + centerHeight;
            double minorRadiusHeightCircleRadius = uniContacts[i]->mu * minorRadiusHeight;
            double minorRadius = sqrt(minorRadiusHeightCircleRadius*minorRadiusHeightCircleRadius - majorRadiusBottomOffset*majorRadiusBottomOffset);

            /// 2d coordinate transformation
            Eigen::Matrix<double, 2, 3> rotation;
            rotation.row(0) = majorAxis[i].transpose();
            rotation.row(1) = minorAxis[i].transpose();
            Vector2d zeroVelVector = rotation * (uniContacts[i]->impulse - center);
            Matrix2d newDistanceMetricTensor = rotation * (inertiaInv[i][i] * rotation.transpose());

            /// new 2d transformation using distance metric
            Matrix2d L, Linv;
            cholesky2d(newDistanceMetricTensor, L, Linv);
            Eigen::DiagonalMatrix<double, 2> elipseM(1.0/majorRadius/majorRadius, 1.0/minorRadius/minorRadius);
            Matrix2d newElipseM = Linv * elipseM * Linv.transpose();
            Vector2d newZeroVelVector = L.transpose() * zeroVelVector;

          }

          uniContacts[i]->impulse = originalImpulse * oneMinusAlpha + uniContacts[i]->impulse * alpha;
          uniContacts[i]->vel = inertiaInv[i][i] * uniContacts[i]->impulse + rest;
          impulseUpdate[i] += uniContacts[i]->impulse;
          velError = std::min(uniContacts[i]->vel(2), 0.0);
          error += impulseUpdate[i].squaredNorm() + velError * velError;
        } else {
          uniContacts[i]->impulse *= oneMinusAlpha;
          error += uniContacts[i]->impulse.squaredNorm();
        }
      }

      if (error < 1e-10) {
//        std::cout<<"counterToBreak "<<counterToBreak<<std::endl;
//        if(slipping) std::cout<<"foot slipping"<<std::endl;
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
      oneMinusAlpha = 1-alpha;
    }

  }

 private:

  inline void projectToFeasibleSet(UnilateralContact *contact,
                                   Vector3d &rest,
                                   double new_z_vel,
                                   int contactID,
                                   double ftan) {
    double tanTheta = -((inertiaInv[contactID][contactID](2, 0) * contact->impulse(0)
        + inertiaInv[contactID][contactID](2, 1) * contact->impulse(1)) / ftan)
        / inertiaInv[contactID][contactID](2, 2);
    double f0 = contact->impulse(2) - tanTheta * ftan;
    double newFtan = f0 / (1 / contact->mu - tanTheta);
    contact->impulse(2) = f0 + tanTheta * newFtan;
    contact->impulse.head(2) *= newFtan / ftan;
  }

  inline bool signOf(const Vector3d &v1, const Vector3d &v2) {
    return signbit(v1(0) * v2(1)) - (v1(1) * v2(0));
  }

  double contactEnergyFcn(double newtonStepRatio) {
    uniContacts_->at(currentContact)->impulse = oldImpulse + newtonStepRatio * angleStep;
    projectToFeasibleSet(uniContacts_->at(currentContact),
                         rest,
                         0.0,
                         currentContact,
                         uniContacts_->at(currentContact)->impulse.head(2).norm());
    xyVel = topInertiaInv[currentContact] * uniContacts_->at(currentContact)->impulse + rest.head(2);
    return xyVel.dot(xyInertia[currentContact] * xyVel);
  }

  inline void cholesky3d(Matrix3d A, Matrix3d L, Matrix3d Linv) {
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < (i+1); j++) {
        double s = 0;
        for (int k = 0; k < j; k++)
          s += L(i, k) * L(j, k);
        L(i, j) = (i == j) ? sqrt(A(i, i) - s) : (1.0 / L(j, j) * (A(i, j) - s));
      }

    /// now finding inverse
    Linv(0,0) = 1.0 / L(0,0);
    Linv(1,1) = 1.0 / L(1,1);
    Linv(2,2) = 1.0 / L(2,2);
    Linv(1,0) = -L(1,0) * Linv(1,1) / L(0,0);
    Linv(2,1) = -Linv(2,2) * L(2,1) / L(1,1);
    Linv(2,0) = (Linv(2,1) * L(1,1) + Linv(2,2) * L(2,1)) / -L(0,1);
  }

  inline void cholesky2d(Matrix2d A, Matrix2d L, Matrix2d Linv) {
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < i+1; j++) {
        double s = 0;
        for (int k = 0; k < j; k++)
          s += L(i, k) * L(j, k);
        L(i, j) = (i == j) ? sqrt(A(i, i) - s) : (1.0 / L(j, j) * (A(i, j) - s));
      }
    L(0,1) = 0.0;

    /// now finding inverse
    Linv(0,0) = 1.0 / L(0,0);
    Linv(1,1) = 1.0 / L(1,1);
    Linv(1,0) = L(1,0) * Linv(1,1) / -L(0,0);
    Linv(0,1) = 0.0;
  }


  Eigen::Matrix3d eye3;
  std::vector<std::vector<Matrix3d> > inertiaInv;
  std::vector<Matrix3d> inertia;
  Vector3d rest, newVel, originalImpulse;
  Vector2d xyVel;
  std::vector<Eigen::Matrix2d> xyInertia;
  std::vector<Eigen::Matrix<double, 2, 3> > topInertiaInv;

  double new_z_vel;
  double alpha, oneMinusAlpha;
  double alpha_low = 0.1;
  double alpha_decay = 0.99;
  double ex;
  double ey;
  double w0exPw1ey;
  double ftanProj;
  Vector3d normToCone, newNormToCone;
  double velError;
  double ftan;

  /// slippage related
  Vector3d direction, gradient, newdirection, angleStep;
  double gradient1D, newgradient1D;
  Vector3d oldImpulse;
  int currentContact;
  std::vector<UnilateralContact *> *uniContacts_;
  std::function<double(double)> fcn_;
  RAI::Utils::Graph::FigProp2D energyFP;
  Eigen::VectorXd angles_plot, energy_plot;


  /// analytic solution
  std::vector<Vector3d> unitNormal;
  std::vector<Vector3d> majorAxis, minorAxis;
  std::vector<double> slope2dMajor;
  std::vector<double> slope2dMinor;

};

}
}

#endif //RAICONTACT2
