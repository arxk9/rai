#include <iostream>
#include <sstream>
#include <functionApproximator/tensorflow/Qfunction_TensorFlow.hpp>
#include <glog/logging.h>

using std::cout;
using std::endl;

int main(int argc, char* argv[]){

  std::ostringstream logPath;
  logPath << getenv("RAI_ROOT") << "/logsNplots/logs/poleBalancingDDDPG";

  FLAGS_alsologtostderr = 1;
  google::SetLogDestination(google::INFO, logPath.str().c_str());
  google::InitGoogleLogging(argv[0]);

  LOG_IF(INFO, true) << "hallo" << endl;

}