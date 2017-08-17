//
// Created by jemin on 10.08.16.
//

#include "RAI_logger/RAI_logger_ToInclude.hpp"

using RAI::Utils::logger;

int main (){

  logger->addVariableToLog(8, "y=x", "testing");

  float data[200];

  for(int i=0; i<200; i+=2) {
    data[i] = float(i);
    data[i+1] = float(i+1);
  }

  for(int i=0; i<100; i+=2)
    logger->appendData("y=x", data+i);

  logger->appendData("y=x", 2,3,4,5,6,7, data);
  logger->appendData("y=x", 2,3,4,5,6,7, data);
  logger->appendData("y=x", 2, 3, 4, 5, 6, 7, 8, 9);

  logger->write2File("testing_file");

}