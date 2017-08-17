//
// Created by jhwangbo on 27.09.16.
//

#include <functional>
#include <Eigen/Core>
#include "math/ConjugateGradient.hpp"
#include <iostream>
#include <RAI_timer/RAI_timer_ToInclude.hpp>
#include "rai/RAI_core"

typedef Eigen::VectorXd Vector;

/////////// A = [1 2; 3 4], b=[1;2], find x that satisfies Ax=b
Eigen::MatrixXd A(4, 5636), M(4,4);

void eval(Vector& x, Vector& Ax){
  Ax = A.transpose()*(M*(A * x));
}

int main(){

  RAI_init();
  Vector b(5636), sol(5636);
  b.setRandom();
  A.setRandom();
  M.setRandom();
  std::function<void(Vector&, Vector&)> fcn = std::bind(&eval, std::placeholders::_1, std::placeholders::_2);
  for(int i=0; i<10; i++) {
    RAI::Utils::timer->startTimer("conjugate gradient");
    RAI::Math::conjugateGradient(fcn, b, 10, 1e-10, sol);
    RAI::Utils::timer->stopTimer("conjugate gradient");
  }
  std::cout<<"solution "<<std::endl<<sol<<std::endl;

}