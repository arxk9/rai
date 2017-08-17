//
// Created by jhwangbo on 30.11.16.
//

#ifndef RAI_QUADROTORCONTROL_HPP
#define RAI_QUADROTORCONTROL_HPP

// custom inclusion- Modify for your task
#include "rai/tasks/common/Task.hpp"
#include "math/RandomNumberGenerator.hpp"
#include "TypeDef.hpp"
#include "math/inverseUsingCholesky.hpp"
#include "math/RAI_math.hpp"
#include "enumeration.hpp"
#include <rai/RAI_core>
#include "RAI_graphics.hpp"

namespace RAI {
namespace Task {

constexpr int StateDim = 18;
constexpr int ActionDim = 4;
constexpr int CommandDim = 0;

template<typename Dtype>
class QuadrotorControl : public Task<Dtype,
                                     StateDim,
                                     ActionDim,
                                     CommandDim> {
 public:
  using TaskBase = Task<Dtype, StateDim, ActionDim, CommandDim>;
  typedef typename TaskBase::Action Action;
  typedef typename TaskBase::ActionBatch ActionBatch;
  typedef typename TaskBase::State State;
  typedef typename TaskBase::StateBatch StateBatch;
  typedef typename TaskBase::Command Command;
  typedef typename TaskBase::VectorXD VectorXD;
  typedef typename TaskBase::MatrixXD MatrixXD;
  typedef typename TaskBase::JacobianStateResAct MatrixJacobian;
  typedef typename TaskBase::JacobianCostResAct MatrixJacobianCostResAct;
  typedef typename Eigen::Matrix<double, 7, 1> GeneralizedCoordinate;
  typedef typename Eigen::Matrix<double, 6, 1> GeneralizedVelocity;
  typedef typename Eigen::Matrix<double, 6, 1> GeneralizedAcceleration;

  QuadrotorControl() {

    //// set default parameters
    this->valueAtTermination_ = 1.5;
    this->discountFactor_ = 0.99;
    this->timeLimit_ = 15.0;
    this->controlUpdate_dt_ = 0.01;
    gravity_ << 0.0, 0.0, -9.81;

    Eigen::Vector3d diagonalInertia;
//    diagonalInertia << 0.0105, 0.0105, 0.018;
    diagonalInertia << 0.007, 0.007, 0.012;

    inertia_ = diagonalInertia.asDiagonal();
    cholInv(inertia_, inertiaInv_);
    comLocation_ << 0.0, 0.0, -0.05;

    /////// scale //////
    actionScale_ = 2.0;
    orientationScale_ = 1.0;
    positionScale_ = 0.5;
    angVelScale_ = 0.15;
    linVelScale_ = 0.5;

    /////// adding constraints////////////////////
    State upperStateBound, lowerStateBound;
    upperStateBound << 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 5, 5, 5, 6, 6, 6;
    lowerStateBound = -upperStateBound;
    this->setBoxConstraints(lowerStateBound, upperStateBound);
    transsThrust2GenForce << 0, 0, length_, -length_,
        -length_, length_, 0, 0,
        dragCoeff_, dragCoeff_, -dragCoeff_, -dragCoeff_,
        1, 1, 1, 1;
    transsThrust2GenForceInv = transsThrust2GenForce.inverse();
    targetPosition.setZero();
  }

  ~QuadrotorControl() {
  }

  void step(const Action &action_t,
            State &state_tp1,
            TerminationType &termType,
            Dtype &costOUT) {
    Action actionGenForce;
    actionGenForce = transsThrust2GenForce * actionScale_ * action_t;
    B_torque = actionGenForce.segment(0,3);
    B_force << 0.0, 0.0, actionGenForce(3);
    orientation = q_.head(4);
    double angle = 2.0 * std::acos(q_(0));
    position = q_.tail(3);
    R_ = Math::MathFunc::quatToRotMat(orientation);
    v_I_ = u_.tail(3);
    w_I_ = u_.head(3);
    double kp_rot = -0.2, kd_rot = -0.06;
    Torque fbTorque_b;
    if(angle > 1e-6)
      fbTorque_b = kp_rot * angle * (R_.transpose() * orientation.tail(3))
          / std::sin(angle) + kd_rot * (R_.transpose() * u_.head(3));
    else
      fbTorque_b = kd_rot * (R_.transpose() * u_.head(3));
    fbTorque_b(2) = fbTorque_b(2) * 0.15;
    B_torque += fbTorque_b;
    B_force(2) += mass_ * 9.81;

    Action genForce; genForce<<B_torque, B_force(2);
    Action thrust = transsThrust2GenForceInv * genForce;

    thrust = thrust.array().cwiseMax(1e-8);
    genForce = transsThrust2GenForce * thrust;
    B_torque = genForce.segment(0,3);
    B_force(2) = genForce(3);

    du_.tail(3) = (R_ * B_force) / mass_ + gravity_;
    w_B_ = R_.transpose() * u_.head(3);

    du_.head(3) = R_ * (inertiaInv_ * (B_torque - w_B_.cross(inertia_ * w_B_)));
    u_ += du_ * this->controlUpdate_dt_;
    w_I_ = u_.head(3);
    w_IXdt_ = w_I_ * this->controlUpdate_dt_;
    orientation = Math::MathFunc::boxplusI_Frame(orientation, w_IXdt_);
    Math::MathFunc::normalizeQuat(orientation);
    q_.head(4) = orientation;
    q_.tail(3) = q_.tail(3) + u_.tail(3) * this->controlUpdate_dt_;
    w_B_ = R_.transpose() * u_.head(3);

    if ( std::isnan(orientation.norm()) ) {
      std::cout << "u_ " << u_.transpose() << std::endl;
      std::cout << "du_ " << du_.transpose() << std::endl;
      std::cout << "B_torque " << B_torque.transpose() << std::endl;
      std::cout << "orientation " << orientation.transpose() << std::endl;
      std::cout << "state_tp1 " << q_.transpose() << std::endl;
      std::cout << "action_t " << action_t.transpose() << std::endl;
    }

    u_(0) = clip(u_(0), -20.0, 20.0);
    u_(1) = clip(u_(1), -20.0, 20.0);
    u_(2) = clip(u_(2), -20.0, 20.0);
    u_(3) = clip(u_(3), -5.0, 5.0);
    u_(4) = clip(u_(4), -5.0, 5.0);
    u_(5) = clip(u_(5), -5.0, 5.0);

    getState(state_tp1);

//    costOUT = 0.004 * q_.tail(3).norm() +
//        0.0002 * action_t.norm() +
//        0.0003 * u_.head(3).norm() +
//        0.0005 * u_.tail(3).norm();

    costOUT = 0.004 * std::sqrt(q_.tail(3).norm()) +
        0.00005 * action_t.norm() +
        0.00005 * u_.head(3).norm() +
        0.00005 * u_.tail(3).norm();


//    std::cout << "distance cost " << 0.004 * q_.tail(3).squaredNorm() << std::endl;
//    std::cout << "actuation cost " << 0.0002 * action_t.squaredNorm() << std::endl;
//    std::cout << "angular velocity cost " << 0.0003 * u_.head(3).squaredNorm() << std::endl;
//    std::cout << "orientation cost " << 0.0005 * acos(q_(0)) * acos(q_(0)) << std::endl;
//    if (this->isViolatingBoxConstraint(state_tp1))
//      termType = TerminationType::timeout;

    // visualization
    if (this->visualization_ON_) {
      std::cout<<"q_ "<<q_.transpose()<<std::endl;
      std::cout<<"thrust "<<thrust.transpose()<<std::endl;

      visualizationTime += this->dt();
//      if(visualizationTime > 1.0/25.0) {
        visualizationTime -= 1.0/25.0;
        Utils::timer->startTimer("visualization");
        graphics.startGraphics();
        quadrotor.draw(position, orientation);
        target.draw(targetPosition);
        graphics.endGraphics();
        usleep(this->controlUpdate_dt_ * 1e6 *1.0);
        Utils::timer->stopTimer("visualization");
//      }
    }
  }

  void stepSim(const Action &action_t) {
    Action thrust = action_t.array().square() * 8.5486e-6;
    thrust = thrust.array().cwiseMax(1e-8);
    Eigen::Matrix<Dtype, 4, 1> genforce;
    genforce = transsThrust2GenForce * thrust;
    B_torque = genforce.segment(0,3);
    B_force << 0.0, 0.0, genforce(3);

    orientation = q_.head(4);
    double angle = 2.0 * std::acos(q_(0));
    position = q_.tail(3);
    R_ = Math::MathFunc::quatToRotMat(orientation);

//    std::cout<<"genForce "<<genForce.transpose()<<std::endl;
//    std::cout<<"angle "<<angle<<std::endl;
//    std::cout<<"orientation.tail(3) / std::sin(angle) "<< orientation.tail(3) / std::sin(angle) <<std::endl;
//    std::cout<<"proportioanl part "<< kp_rot * angle * ( R_.transpose() * orientation.tail(3) ) / std::sin(angle) <<std::endl;
//    std::cout<<"diff part "<< kd_rot * ( R_.transpose() * u_.head(3) ) <<std::endl;

    du_.tail(3) = (R_ * B_force) / mass_ + gravity_;
    w_B_ = R_.transpose() * u_.head(3);

    /// compute in body coordinate and transform it to world coordinate
//    B_torque += comLocation_.cross(mass_ * R_.transpose() * gravity_);
    du_.head(3) = R_ * (inertiaInv_ * (B_torque - w_B_.cross(inertia_ * w_B_)));

    u_ += du_ * this->controlUpdate_dt_;

    w_I_ = u_.head(3);
    w_IXdt_ = w_I_ * this->controlUpdate_dt_;
    orientation = Math::MathFunc::boxplusI_Frame(orientation, w_IXdt_);
    Math::MathFunc::normalizeQuat(orientation);
    q_.head(4) = orientation;
    q_.tail(3) = q_.tail(3) + u_.tail(3) * this->controlUpdate_dt_;
    w_B_ = R_.transpose() * u_.head(3);

    if ( std::isnan(orientation.norm()) ) {
      std::cout << "u_ " << u_.transpose() << std::endl;
      std::cout << "du_ " << du_.transpose() << std::endl;
      std::cout << "B_torque " << B_torque.transpose() << std::endl;
      std::cout << "orientation " << orientation.transpose() << std::endl;
      std::cout << "state_tp1 " << q_.transpose() << std::endl;
      std::cout << "action_t " << action_t.transpose() << std::endl;
    }

    // visualization
    if (this->visualization_ON_) {
      visualizationTime += this->dt();
      if(visualizationTime > 1.0/25.0) {
        visualizationTime -= 1.0/25.0;
        Utils::timer->startTimer("visualization");
        graphics.startGraphics();
        quadrotor.draw(position, orientation);
        target.draw(targetPosition);
        graphics.endGraphics();
        Utils::timer->stopTimer("visualization");
      }
    }
  }


  void changeTarget(Position& position){
    targetPosition = position;
  }

  bool isTerminalState(State &state) { return false; }

  void init() {
    /// initial state is random
    double oriF[4], posiF[3], angVelF[3], linVelF[3];
    rn_.template sampleOnUnitSphere<4>(oriF);
    rn_.template sampleVectorInNormalUniform<3>(posiF);
    rn_.template sampleVectorInNormalUniform<3>(angVelF);
    rn_.template sampleVectorInNormalUniform<3>(linVelF);
    Quaternion orientation;
    Position position;
    AngularVelocity angularVelocity;
    LinearVelocity linearVelocity;

    orientation << double(std::abs(oriF[0])), double(oriF[1]), double(oriF[2]), double(oriF[3]);
    RAI::Math::MathFunc::normalizeQuat(orientation);
    position << double(posiF[0])*2., double(posiF[1])*2., double(posiF[2])*2.;
    angularVelocity << double(angVelF[0]), double(angVelF[1]), double(angVelF[2]);
    linearVelocity << double(linVelF[0]), double(linVelF[1]), double(linVelF[2]);

    q_ << orientation, position;
    u_ << angularVelocity, linearVelocity;
  }

  void translate(Position& position) {
    q_.segment(4,3) += position;
  }

  void getInitialState(State &state) {
    init();
    getState(state);
  }

  void setInitialState(const State &in) {
    LOG(FATAL) << "The initial state is random. No need to set it" << std::endl;
  }

  void initTo(const State &state) {
    State stateT = state;
    R_.col(0) = stateT.segment(0, 3);
    R_.col(1) = stateT.segment(3, 3);
    R_.col(2) = stateT.segment(6, 3);
    orientation = Math::MathFunc::rotMatToQuat(R_);
    q_.head(4) = orientation;
    q_.tail(3) = state.segment(9, 3) / positionScale_;
    u_.head(3) = state.segment(12, 3) / angVelScale_;
    u_.tail(3) = state.segment(15, 3) / linVelScale_;
  }

  void getState(State &state) {
    LOG_IF(FATAL, std::isnan(q_.head(4).norm())) << "simulation unstable";
    orientation = q_.head(4);
    Math::MathFunc::normalizeQuat(orientation);
    R_ = Math::MathFunc::quatToRotMat(orientation);
    state << R_.col(0), R_.col(1), R_.col(2),
        q_.tail(3) * positionScale_,
        u_.head(3) * angVelScale_,
        u_.tail(3) * linVelScale_;
  };

  // Misc implementations
  void getGradientStateResAct(const State &stateIN,
                              const Action &actionIN,
                              MatrixJacobian &gradientOUT) {
    LOG(FATAL) << "To do!" << std::endl;
  };

  void getGradientCostResAct(const State &stateIN,
                             const Action &actionIN,
                             MatrixJacobianCostResAct &gradientOUT) {
    LOG(FATAL) << "To do!" << std::endl;
  }

  void getOrientation(Quaternion &quat){
    quat = orientation;
  }

  void getPosition(Position &posi){
    posi = q_.tail(3);
  }

  void getLinvel(LinearVelocity &linvel){
    linvel = u_.tail(3);
  }

  void getAngvel(AngularVelocity &angvel){
    angvel = u_.head(3);
  }

 private:

  double clip(double input, double lower, double upper) {
    input = (input > lower) * input + !(input > lower) * lower;
    return (input < upper) * input + !(input < upper) * upper;
  }

  template<typename T>
  inline double sgn(T val) {
    return double((T(0) < val) - (val < T(0)));
  }

  GeneralizedCoordinate q_; // generalized state and velocity
  GeneralizedVelocity u_;
  GeneralizedAcceleration du_;
  RotationMatrix R_;
  LinearAcceleration gravity_;

  double orientationScale_, positionScale_, angVelScale_, linVelScale_;
  double actionScale_;
  /// robot parameters
  double length_ = 0.17;
  Position comLocation_;
  double dragCoeff_ = 0.016;
  double mass_ = 0.665;
  Inertia inertia_;
  Inertia inertiaInv_;
  Quaternion orientation;
  Position position;
  Force B_force;
  Torque B_torque;
  AngularVelocity w_I_, w_B_;
  LinearVelocity v_I_;
  RandomNumberGenerator<Dtype> rn_;
  EulerVector w_IXdt_;
  static Graphics::RAI_graphics graphics;
  static Graphics::Objects::Quadrotor quadrotor;
  static Graphics::Objects::Sphere target;
  static Position targetPosition;
  double visualizationTime = 0;

  Eigen::Matrix4d transsThrust2GenForce;
  Eigen::Matrix4d transsThrust2GenForceInv;

};

}
} /// namespaces

template<typename Dtype>
RAI::Graphics::RAI_graphics RAI::Task::QuadrotorControl<Dtype>::graphics(
"quad simulation", {800, 800}, {0, 0}, RAI::Graphics::RAI_graphics::RAI_3D, {7.0, 5.0} );

template<typename Dtype>
RAI::Graphics::Objects::Quadrotor RAI::Task::QuadrotorControl<Dtype>::quadrotor( { 0.1, 0.1, 0.8 } );
template<typename Dtype>
RAI::Graphics::Objects::Sphere RAI::Task::QuadrotorControl<Dtype>::target(0.05, { 0.9, 0.1, 0.1 } );
template<typename Dtype>
RAI::Position RAI::Task::QuadrotorControl<Dtype>::targetPosition;
#endif //RAI_QUADROTORCONTROL_HPP