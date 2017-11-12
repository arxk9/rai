//
// Created by joonho on 11/12/17.
//

//
// Created by joonho on 23.03.17.
//


#include <iostream>
#include "rai/function/tensorflow/common/ParameterizedFunction_TensorFlow.hpp"
#include "rai/function/tensorflow/Qfunction_TensorFlow.hpp"
#include "rai/function/tensorflow/DeterministicPolicy_TensorFlow.hpp"
#include "rai/function/tensorflow/StochasticPolicy_TensorFlow.hpp"

#include "rai/function/tensorflow/ValueFunction_TensorFlow.hpp"
#include "rai/function/common/Policy.hpp"
#include "rai/common/math/RAI_math.hpp"

#include <math.h>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <functional>
#include <rai/function/tensorflow/RecurrentStochasticPolicy_TensorFlow.hpp>

#include "rai/RAI_core"
#include "rai/RAI_Tensor.hpp"
#include <rai/algorithm/common/LearningData.hpp>



using std::cout;
using std::endl;
using std::cin;
const int ActionDim = 2;
const int StateDim = 3;
using Dtype = float;

using NormNoise = rai::Noise::NormalDistributionNoise<Dtype, ActionDim>;
using NoiseCov = Eigen::Matrix<Dtype, ActionDim, ActionDim>;

using PolicyBase = rai::FuncApprox::Policy<Dtype, StateDim, ActionDim>;
using RnnPolicy = rai::FuncApprox::RecurrentStochasticPolicy_TensorFlow<Dtype, StateDim, ActionDim>;

using MatrixXD = Eigen::Matrix<Dtype, -1, -1>;
using VectorXD = Eigen::Matrix<Dtype, -1, 1>;
using Advantages = Eigen::Matrix<Dtype, 1, Eigen::Dynamic>;
typedef typename PolicyBase::State State;
typedef typename PolicyBase::StateBatch StateBatch;
typedef typename PolicyBase::Action Action;
typedef typename PolicyBase::ActionBatch ActionBatch;
typedef typename PolicyBase::JacobianWRTparam JacobianWRTparam;
typedef typename rai::Algorithm::LearningData<Dtype, StateDim, ActionDim>::tensorBatch TensorBatch;

using namespace rai;
//
//Dtype sample(double dummy) {
//  static std::mt19937 rng;
//  static std::normal_distribution<Dtype> nd(training_mean, sqrt(training_variance));
//  return nd(rng);
//}

int main() {
  RAI_init();
  bool teststdev = false;
  bool testgradient = false;
  const int sampleN = 5; 
  int Batsize = 100;
  int len = 100;

  RnnPolicy policy("cpu", "GRUMLP_TBPTT", "tanh 3 16 / 16 16 1", 0.001);

  rai::Tensor<Dtype,3> states("state");
  states.resize(StateDim,len,Batsize);
  states.setZero();
  
  TensorBatch test_bat;
  
  test_bat.states = states;

};