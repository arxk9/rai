//
// Created by jhwangbo on 05.02.17.
//

#include <rai/RAI_core>
#include "rai/tasks/quadrotor/QuadrotorControl.hpp"
#include "rai/tasks/quadrotor/QuadSimulation.hpp"
#include "rai/tasks/quadrotor/MLP_QuadControl.hpp"
#include "rai/function/tensorflow/DeterministicPolicy_TensorFlow.hpp"
#include <termios.h>

using rai::Task::ActionDim;
using rai::Task::StateDim;

char getkey() {
  char buf = 0;
  struct termios old = {0};
  if (tcgetattr(0, &old) < 0)
    perror("tcsetattr()");
  old.c_lflag &= ~ICANON;
  old.c_lflag &= ~ECHO;
  old.c_cc[VMIN] = 1;
  old.c_cc[VTIME] = 0;
  if (tcsetattr(0, TCSANOW, &old) < 0)
    perror("tcsetattr ICANON");
  if (read(0, &buf, 1) < 0)
    perror("read()");
  old.c_lflag |= ICANON;
  old.c_lflag |= ECHO;
  if (tcsetattr(0, TCSADRAIN, &old) < 0)
    perror("tcsetattr ~ICANON");
  return (buf);
}

int keyboardHit(void) {
  struct timeval tv;
  fd_set rdfs;
  tv.tv_sec = 0;
  tv.tv_usec = 0;
  FD_ZERO(&rdfs);
  FD_SET (STDIN_FILENO, &rdfs);
  select(STDIN_FILENO + 1, &rdfs, NULL, NULL, &tv);
  return FD_ISSET(STDIN_FILENO, &rdfs);
}

int main() {
  RAI_init();

  rai::Task::QuadSimulation<double> sim;
  rai::MLP_QuadControl mlp("TensorflowPB_Quad/2017-02-19-14-08-45/policy_200.txt");
//  rai::MLP_QuadControl mlp("TensorflowPB_Quad/2017-02-12-00-39-25/policy_1550.txt");

//  rai::FuncApprox::Policy_TensorFlow<double, 18, 4> tf("TensorflowPB_Quad/2017-02-19-14-08-45/policy_2l.pb",1e-3);
//  tf.loadParam("TensorflowPB_Quad/2017-02-19-14-08-45/policy_3100.txt");

  int timeStep=0;
  while (true) {
//    sim.loop(tf);
    sim.loop(mlp);
    timeStep++;

//    if(timeStep * 0.01 > 3.0)
//      break;

    if (keyboardHit() == true)
      if (getkey() == '\n')
        sim.randomInit();
  }
}