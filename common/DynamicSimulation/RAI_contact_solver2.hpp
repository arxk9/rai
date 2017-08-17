//
// Created by jhwangbo on 3/12/17.
//

#ifndef RAI_RAICONTACT2_HPP
#define RAI_RAICONTACT2_HPP

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

class RAI_contact_solver2 {

  typedef Eigen::Vector3d Vector3d;
  typedef Eigen::Vector2d Vector2d;
  typedef Eigen::Matrix3d Matrix3d;
  typedef Eigen::VectorXd VectorXd;

 public:
  RAI_contact_solver2() {
    /////////////////////// Plotting properties ////////////////////////
//    energyFP.title = "energy vs angle";
//    energyFP.xlabel = "angles";
//    energyFP.ylabel = "energy";
//
//    energy_plot.resize(360);
//    angles_plot.resize(360);
//
//    for(int dataID = 0; dataID < 360; dataID++)
//      angles_plot(dataID) = dataID / 180.0 * M_PI - M_PI;
//    Utils::logger->addVariableToLog(1, "ctb", "counter-To-break of cs2");
  }

  ~RAI_contact_solver2() {}

  void solve(std::vector<UnilateralContact *> &uniContacts,
             const MatrixXd &M_inv, const VectorXd &tauStar) {

//    Utils::timer->startTimer("matrix initialization");
    uniContacts_ = &uniContacts;

    unsigned contactN = uniContacts.size();
    unsigned stateDim = M_inv.cols();
    alpha = 0.9;
    oneMinusAlpha = 1-alpha;

    std::vector<Vector3d> c(contactN);
    std::vector<Vector3d> impulseUpdate(contactN);

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
      cholInv(inertiaInv[i][i], inertia[i]);
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

//        Utils::timer->startTimer("rest comp");
        /// initially compute velcoties
        rest = c[i];
        for (unsigned j = 0; j < contactN; j++)
          if (i != j) rest += inertiaInv[i][j] * uniContacts[j]->impulse;
//        Utils::timer->stopTimer("rest comp");

        /// if it is on ground
        if (rest(2) < 0) {
          impulseUpdate[i] = -uniContacts[i]->impulse; /// storing previous impulse for impulse update computation.
          /// This is only to check the terminal condition
          originalImpulse = uniContacts[i]->impulse;
          newVel.setZero();
          uniContacts[i]->impulse = inertia[i] * (newVel - rest);
          ftan = uniContacts[i]->impulse.head(2).norm();

          if (uniContacts[i]->impulse(2) * uniContacts[i]->mu > ftan) {
            /// if no slip take a step toward the non-slip impulse
          } else {
            oldImpulse = uniContacts[i]->impulse;
            projectToFeasibleSet(uniContacts[i], rest, 0.0, i, ftan);
//            float firstMinimum = atan2(uniContacts[i]->impulse(1), uniContacts[i]->impulse(0));
//            float firstEnergy = 0;
//            {
//              Vector3d vel_test;
//              vel_test = inertiaInv[i][i] * uniContacts[i]->impulse + rest;
//              firstEnergy = vel_test.dot(inertia[i] * vel_test);
//            }

//            Utils::timer->startTimer("angle stepping");
            normToCone << uniContacts[i]->impulse(0), uniContacts[i]->impulse(1), (-uniContacts[i]->mu
                * uniContacts[i]->mu) * uniContacts[i]->impulse(2);
            Vector3d zeroTanNormalImpulse(0,0,-rest(2) / inertiaInv[i][i](2,2));

            /// if it is going to slip
            newdirection = inertiaInv[i][i].row(2).transpose().cross(normToCone);
            newgradient1D = newdirection.dot(inertiaInv[i][i] * uniContacts[i]->impulse + rest);
            double negativeMuSquared = -uniContacts[i]->mu * uniContacts[i]->mu;
            ftan = uniContacts[i]->impulse.head(2).norm();

            /// fixed angle stepping
            double multiplier = -0.05 *sgn(newgradient1D) * std::min((oldImpulse-uniContacts[i]->impulse).norm(), (uniContacts[i]->impulse - zeroTanNormalImpulse).norm());
            do {
              gradient1D = newgradient1D;
              direction = newdirection;
              angleStep = multiplier* direction;
              multiplier *= 3;
              oldImpulse = uniContacts[i]->impulse;
              uniContacts[i]->impulse = oldImpulse + angleStep;
              ftan = uniContacts[i]->impulse.head(2).norm();
              projectToFeasibleSet(uniContacts[i], rest, 0.0, i, ftan);
              normToCone << uniContacts[i]->impulse(0), uniContacts[i]->impulse(1), negativeMuSquared * uniContacts[i]->impulse(2);
              newdirection = inertiaInv[i][i].row(2).transpose().cross(normToCone);
              newgradient1D = newdirection.dot(inertiaInv[i][i] * uniContacts[i]->impulse + rest);
              direction = newdirection;
            } while (gradient1D * newgradient1D > 0);
//            Utils::timer->stopTimer("angle stepping");


//            float angleMinimum = atan2(uniContacts[i]->impulse(1), uniContacts[i]->impulse(0));
//            float angleEnergy = 0;
//            {
//              Vector3d vel_test;
//              vel_test = inertiaInv[i][i] * uniContacts[i]->impulse + rest;
//              angleEnergy = vel_test.dot(inertia[i] * vel_test);
//            }

//            Utils::timer->startTimer("bisection");
            nextImpulse = uniContacts[i]->impulse;
            /// bisection method
            for(int interN = 0; interN < 22; interN++) {
              uniContacts[i]->impulse = (oldImpulse + nextImpulse) / 2.0;
              projectToFeasibleSet(uniContacts[i], rest, 0.0, i, uniContacts[i]->impulse.head(2).norm());
              normToCone << uniContacts[i]->impulse(0), uniContacts[i]->impulse(1), -uniContacts[i]->mu
                  * uniContacts[i]->mu * uniContacts[i]->impulse(2);
              newdirection = inertiaInv[i][i].row(2).transpose().cross(normToCone);
              newgradient1D = newdirection.dot(inertiaInv[i][i] * uniContacts[i]->impulse + rest);
              if (gradient1D * newgradient1D > 0)
                oldImpulse = uniContacts[i]->impulse;
              else
                nextImpulse = uniContacts[i]->impulse;
            }
//            Utils::timer->stopTimer("bisection");

//            float bsMinimum = atan2(uniContacts[i]->impulse(1), uniContacts[i]->impulse(0));
//            float bsEnergy = 0;
//            {
//              Vector3d vel_test;
//              vel_test = inertiaInv[i][i] *Utils::logger->appendData("ctb", counterToBreak); uniContacts[i]->impulse + rest;
//              bsEnergy = vel_test.dot(inertia[i] * vel_test);
//            }

//            if(bsEnergy > firstEnergy) {
//              ////// plotting /////////
//              for(int dataID = 0; dataID < 360; dataID++) {
//                Vector3d force_test;
//                double theta = angles_plot(dataID);
//                force_test(2) = -rest(2) / (inertiaInv[i][i](2,0) * uniContacts[i]->mu * std::cos(theta) +
//                    inertiaInv[i][i](2,1) * uniContacts[i]->mu * sin(theta) + inertiaInv[i][i](2,2));
//                force_test(0) = uniContacts[i]->mu * force_test(2) * cos(theta);
//                force_test(1) = uniContacts[i]->mu * force_test(2) * sin(theta);
//                Vector3d vel_test;
//                vel_test = inertiaInv[i][i] * force_test + rest;
//                energy_plot(dataID) = vel_test.dot(inertia[i] * vel_test);
//              }
//
//              Utils::graph->figure(0, energyFP);
//              Utils::graph->appendData(0,angles_plot.data(), energy_plot.data(), 360, "energy");
//              Utils::graph->appendData(0, &angleMinimum, &angleEnergy, 1, Utils::Graph::PlotMethods2D::points, "angle minimum", "ps 2 pt 7");
//              Utils::graph->appendData(0, &firstMinimum, &firstEnergy, 1, Utils::Graph::PlotMethods2D::points, "projection", "ps 2 pt 7");
//              Utils::graph->appendData(0, &bsMinimum, &bsEnergy, 1, Utils::Graph::PlotMethods2D::points, "bs minimum", "ps 2 pt 7");
//              Utils::graph->drawFigure(0);
//              std::cout<<"first minimum "<<firstEnergy<<std::endl;
//              std::cout<<"angle minimum "<<angleEnergy<<std::endl;
//              std::cout<<"bs    minimum "<<bsEnergy<<std::endl;
//              Utils::graph->waitForEnter();
//            }

          }

          uniContacts[i]->impulse = originalImpulse * oneMinusAlpha + uniContacts[i]->impulse * alpha;
          velError = std::min(inertiaInv[i][i].row(2) * uniContacts[i]->impulse + rest(2), 0.0);
          impulseUpdate[i] += uniContacts[i]->impulse;
          error += impulseUpdate[i].squaredNorm() + velError * velError;
        } else {
          uniContacts[i]->impulse *= oneMinusAlpha;
          error += uniContacts[i]->impulse.squaredNorm();
        }
      }

      if (error < 1e-12) {
//        std::cout<<counterToBreak<<std::endl;
//        Utils::logger->appendData("ctb", counterToBreak);
        break;
      }

      if (++counterToBreak > 1000) {
//        if(error > 1e-5) {
          std::cout << "the Contact dynamic solver did not converge in time, the error is " << error << std::endl;
          std::cout << "the Contact force is " << uniContacts[0]->impulse.transpose() << std::endl;
//        }
        break;
      }
      alpha = (alpha - alpha_low) * alpha_decay + alpha_low;
      oneMinusAlpha = 1-alpha;
    }
//    Utils::timer->stopTimer("Solver");
  }

 private:

  inline void projectToFeasibleSet(UnilateralContact *contact,
                                   Vector3d &rest,
                                   double new_z_vel,
                                   int contactID,
                                   double ftan) {
    double neg_tanTheta = (inertiaInv[contactID][contactID](2, 0) * contact->impulse(0)
        + inertiaInv[contactID][contactID](2, 1) * contact->impulse(1)) / (ftan
        * inertiaInv[contactID][contactID](2, 2));
    double f0 = contact->impulse(2) + neg_tanTheta * ftan;
    double newFtan = f0 / (1 / contact->mu + neg_tanTheta);
    contact->impulse(2) = f0 - neg_tanTheta * newFtan;
    contact->impulse.head(2) *= newFtan / ftan;
  }

  template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
  }

  std::vector<std::vector<Matrix3d> > inertiaInv;
  std::vector<Matrix3d> inertia;
  Vector3d rest, newVel, originalImpulse;

  double alpha, oneMinusAlpha;
  double alpha_low = 0.1;
  double alpha_decay = 0.99;
  Vector3d normToCone;
  double velError;
  double ftan;

  /// slippage related
  Vector3d direction, gradient, newdirection, angleStep;
  double gradient1D, newgradient1D;
  Vector3d oldImpulse, nextImpulse;
  std::vector<UnilateralContact *> *uniContacts_;
//  RAI::Utils::Graph::FigProp2D energyFP;
//  Eigen::VectorXd angles_plot, energy_plot;

};

}
}

#endif //RAICONTACT2
