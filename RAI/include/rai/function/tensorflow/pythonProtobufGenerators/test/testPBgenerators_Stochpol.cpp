//
// Created by joonho on 23.03.17.
//


#include <iostream>
#include "rai/function/tensorflow/ParameterizedFunction_TensorFlow.hpp"
#include "rai/function/tensorflow/Qfunction_TensorFlow.hpp"
#include "rai/function/tensorflow/DeterministicPolicy_TensorFlow.hpp"
#include "rai/function/tensorflow/StochasticPolicy_TensorFlow.hpp"
#include "rai/function/tensorflow/ValueFunction_TensorFlow.hpp"
#include "rai/function/common/Policy.hpp"
#include <math/RAI_math.hpp>

#include <math.h>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <functional>

#include "rai/RAI_core"

using std::cout;
using std::endl;
using std::cin;
const int ActionDim = 10;
const int StateDim = 20;
using Dtype = float;

using NormNoise = RAI::Noise::NormalDistributionNoise<Dtype, ActionDim>;
using NoiseCov = Eigen::Matrix<Dtype, ActionDim, ActionDim>;

using PolicyBase = RAI::FuncApprox::Policy<Dtype, StateDim, ActionDim>;
using stPolicy = RAI::FuncApprox::StochasticPolicy_TensorFlow<Dtype, StateDim, ActionDim>;

using MatrixXD = Eigen::Matrix<Dtype, -1, -1>;
using VectorXD = Eigen::Matrix<Dtype, -1, 1>;
using Advantages = Eigen::Matrix<Dtype, 1, Eigen::Dynamic>;
typedef typename PolicyBase::State State;
typedef typename PolicyBase::StateBatch StateBatch;
typedef typename PolicyBase::Action Action;
typedef typename PolicyBase::ActionBatch ActionBatch;
typedef typename PolicyBase::JacobianWRTparam JacobianWRTparam;

using namespace RAI;
//
//Dtype sample(double dummy) {
//  static std::mt19937 rng;
//  static std::normal_distribution<Dtype> nd(training_mean, sqrt(training_variance));
//  return nd(rng);
//}

int main() {
  RAI_init();
  bool teststdev = true;
  bool testgradient = false;
  const int sampleN = 200000;
  stPolicy policy("stochpol_3l", "20 10 relu 32 32 32 1e-3", 1e-3, 1);
  stPolicy policy2("Stochastic_pol", "relu",{32, 32, 32}, 1e-3, 1e-3, 1);

  StateBatch stateBatch = StateBatch::Ones(StateDim, sampleN);
  ActionBatch actionBatch = ActionBatch::Ones(ActionDim, sampleN);
  ActionBatch actionBatch2 = ActionBatch::Ones(ActionDim, sampleN);

  ActionBatch actionNoise = ActionBatch::Ones(ActionDim, sampleN) * 0.4;
  ActionBatch actionmean;
  ActionBatch actiontest, actionNew;
  Action Stdev_o, StdevNew;
  Action actionstdv;
  State state = State::Ones(StateDim);
  Action action, action2;

  VectorXD parameter,parameter2, testgrad, testgrad2, temp, fishergrad,fishergrad2;
  MatrixXD test;
  Advantages advs = Eigen::Matrix<Dtype, 1, sampleN>::Ones(1, sampleN);
  Dtype loss, loss2;



 if(teststdev){

  NoiseCov Cov;
//   Cov = NoiseCov::Identity() * Dtype(0.3);

   Cov = NoiseCov::Ones() * 0.5;
   Cov = Cov.array().abs();
  Action Stdev_n;
  NormNoise Normal(Cov);
  Stdev_n = Normal.getCovariance().diagonal();

  std::cout << "New stdev: " << Stdev_n ;
   policy.getStdev(Stdev_o);
   std::cout << "old stdev: " << Stdev_o ;

   policy.setStdev(Stdev_n);

   policy.getStdev(Stdev_o);
   std::cout << "error: " << Stdev_o - Stdev_n ;


}



  parameter.setZero(policy.getLPSize());
  testgrad.Random(policy.getLPSize());
  fishergrad.setZero(policy.getLPSize());
  parameter2.setZero(policy.getLPSize());
  testgrad2.Random(policy.getLPSize());
  fishergrad2.setZero(policy.getLPSize());

  policy.getLP(parameter);
  policy.setLP(parameter);
  policy2.setLP(parameter);

  policy.getLP(parameter);
  policy2.getLP(parameter2);

  LOG(INFO) << "LP diff" << (parameter-parameter2).norm()*100;

  policy.getpg(stateBatch,actionBatch,actionNoise,advs,Stdev_o,testgrad);
  policy2.getpg(stateBatch,actionBatch,actionNoise,advs,Stdev_o,testgrad2);
  LOG(INFO) << "grad diff" << (testgrad-testgrad2).norm()/testgrad.norm()*100;

  policy.getfvp(stateBatch,actionBatch,actionNoise,advs,Stdev_o,testgrad,fishergrad);
  policy2.getfvp(stateBatch,actionBatch,actionNoise,advs,Stdev_o,testgrad2,fishergrad2);

  LOG(INFO) << "fvp diff" << (fishergrad-fishergrad2).norm()/fishergrad2.norm()*100;

  policy.forward(stateBatch,actionBatch);
  policy2.forward(stateBatch,actionBatch2);

  LOG(INFO) << "fwd diff" << (actionBatch-actionBatch2).norm()/actionBatch.norm()*100;

  policy.forward(state,action);
  policy2.forward(state,action2);

  LOG(INFO) << "fwd diff2" << (action-action2).norm()/action.norm()*100;

  Stdev_o = Stdev_o * 0.6;

  policy.setStdev(Stdev_o);
  policy2.setStdev(Stdev_o);

  policy.getStdev(Stdev_o);
  policy2.getStdev(StdevNew);

  LOG(INFO) << "stdev diff" << (Stdev_o-StdevNew).norm()/Stdev_o.norm()*100;

  loss = policy.TRPOloss(stateBatch, actionBatch, actionNoise, advs, Stdev_o);
  loss2 = policy2.TRPOloss(stateBatch, actionBatch, actionNoise, advs, Stdev_o);
  LOG(INFO) << "loss diff" << (loss-loss2)/loss*100 ;
  LOG(INFO) << "Loss : " << loss << endl;

  /////// test gradient
  if (testgradient) {
    VectorXD p_n, p_old, rate;

    Dtype Loss = 0;
    int paramsize;
    VectorXD pgtest, pgtest2;
    paramsize = policy.getLPSize();

    p_n.resize(sampleN);
    p_old.resize(sampleN);
    rate.resize(sampleN);
    policy.getStdev(Stdev_o);
    pgtest.setZero(policy.getLPSize());
    pgtest2.setZero(policy.getLPSize());
    policy.getLP(parameter);

    Action Stdev = Stdev_o.col(0);
    Dtype stepsize = 1e-5;
    VectorXD paramnew(paramsize);
    Dtype loss2;

    policy.getpg(stateBatch, actionBatch, actionNoise, advs, Stdev_o, testgrad);

    for (int i = 0; i < sampleN; i++) {
      Action Temp2 = actionNoise.col(i).array() / Stdev.array();
      p_old[i] = -0.5 * Temp2.transpose() * Temp2 - (Stdev.array().log()).sum() ;//- 0.5 * std::log(2.0 * M_PI) * ActionDim;
    }

    for (int j = 0; j < paramsize; j++) {
      paramnew = parameter;
      paramnew(j) += stepsize;

      policy.setLP(paramnew);
      loss2 = policy.TRPOloss(stateBatch, actionBatch, actionNoise, advs, Stdev_o);

      policy.forward(stateBatch, actionNew);
      policy.getStdev(StdevNew);

      ActionBatch Noise_new = actionBatch - actionNew;
      Action StdevN = StdevNew.col(0);

      Loss = 0;
      for (int i = 0; i < sampleN; i++) {
        Action Temp = Noise_new.col(i).array() / StdevN.array();
        p_n[i] = -0.5 * Temp.transpose() * Temp - (StdevN.array().log()).sum() ;//- 0.5 * std::log(2.0 * M_PI) * ActionDim;
       rate[i] =  std::exp(p_n[i] - p_old[i]);
        Loss += std::exp(p_n[i] - p_old[i]) * advs[i];
      }
      Loss = -Loss / sampleN;
      LOG(INFO) <<j<< " Loss Function Error : " << Loss - loss2;

      ///DEBUG
//      policy.test(stateBatch, actionBatch, actionNoise, advs, Stdev_o, test);
//      cout << p_old - test << endl << endl;

      pgtest(j) = (Loss - loss) / stepsize;
      pgtest2(j) = (loss2 - loss) / stepsize;
    }
    cout <<endl;
    cout << "TF - Numeric error (C++loss) :" << (pgtest - testgrad).norm()/pgtest.norm() << endl;
    cout << "TF - Numeric error (TFloss) :" << (pgtest2 - testgrad).norm()/pgtest.norm();
  }

//  policy.getfvp(stateBatch, actionBatch, actionNoise, advs, Stdev_o,testgrad, fishergrad);
}


//
//logp_n = - 0.5 * tf.reduce_sum(tf.square((OldActionSampled - action_mean) / action_stdev), axis=1) \
//                     - 0.5 * tf.cast(tf.log(2.0 * np.pi),dtype) * action_dim - tf.reduce_sum(tf.log(action_stdev))
//logp_old = - 0.5 * tf.reduce_sum(tf.square((OldActionNoise) / OldStdv), axis=1) \
//                     - 0.5 * tf.cast(tf.log(2.0 * np.pi),dtype) * action_dim - tf.reduce_sum(tf.log(OldStdv))
//
//ratio = tf.exp(logp_n - logp_old, name='rat')
//surr = - tf.reduce_mean(tf.multiply(ratio, advant), name='Surr')