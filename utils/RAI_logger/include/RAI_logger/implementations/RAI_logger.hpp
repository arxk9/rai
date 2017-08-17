//
// Created by jemin on 10.08.16.
// you can store data in memory for logsNplots with gnuplot or sotre in file for logsNplots in matlab
//

#ifndef RAI_DATA_LOGGER_HPP
#define RAI_DATA_LOGGER_HPP
#include <map>
#include "glog/logging.h"
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits.h>
#include "RAI_data.hpp"
#include <iostream>

namespace RAI {
namespace Utils {

class RAI_logger {

 public:
  RAI_logger() {};
  ~RAI_logger() {

    std::ostringstream logPath;
    logPath << log_path_ << "/dataLog.rlog";
    std::ofstream logFile;
    logFile.open(logPath.str().c_str());

    logFile<<"Executable: "<<getExePath()<< "\n\n";

    for (int id = 0; id < data_.size(); id++)
      data_[id]->write2File(logFile);

    for (int id = 0; id < data_.size(); id++)
      delete data_[id];
  };

  /* names are separated by "/". ex) RAI/DDPG/LearningCurves */
  void addVariableToLog(int dim, std::string name, std::string description) {
    data_.push_back(new RAI_data(dim, name, description));
    dataIdx_.insert(std::pair<std::string, int>(name, numberOfData_));
    numberOfData_++;
  }

  template<typename ...Dtype>
  void appendData(std::string name, Dtype... values) {
    data_[dataIdx_[name]]->appendNewData(values...);
  }

  template<typename ...Args, typename Dtype>
  void appendData(std::string name, Args... values, Dtype *data) {
    data_[dataIdx_[name]]->appendNewData(values..., data);
  }

  float const *getData(std::string name, int dimensionIdx) {
    LOG_IF(FATAL, dataIdx_.find(name) == dataIdx_.end())
    << "no such Data available";
    LOG_IF(FATAL, dimensionIdx >= getDimension(name))
    << "Dimension index requested exceeds the dimension of the data";
    return &data_[dataIdx_[name]]->getData()[dimensionIdx][0];
  }

  const std::vector<std::vector<float> > *getData(std::string name) {
    LOG_IF(FATAL, dataIdx_.find(name) == dataIdx_.end())
    << "no such Data available";
    return &data_[dataIdx_[name]]->getData();
  }

  int getDataSize(std::string name) {
    LOG_IF(FATAL, dataIdx_.find(name) == dataIdx_.end())
    << "no such Data available";
    return data_[dataIdx_[name]]->getData()[0].size();
  }

  int getDimension(std::string name) {
    return data_[dataIdx_[name]]->getData().size();
  }

  void dumpData(std::string name) {
    data_[dataIdx_[name]]->dumpData();
  }

  std::string getCurrentDataAndTime() {
    std::ostringstream dataAndTime;
    time_t t = time(0);   // get time now
    struct tm *now = localtime(&t);
    dataAndTime << (now->tm_year + 1900) << '-' << std::setfill('0')
                << std::setw(2) << (now->tm_mon + 1) << '-'
                << std::setw(2) << now->tm_mday << '-' << std::setw(2)
                << now->tm_hour << '-' << std::setw(2)
                << now->tm_min << '-' << std::setw(2) << now->tm_sec;
    return dataAndTime.str().c_str();
  }

  void setLogPath(std::string& path){
    log_path_ = path;
  }

 private:
  std::vector<RAI_data *> data_;
  std::map<std::string, int> dataIdx_;
  int numberOfData_ = 0;
  std::string log_path_;

  std::string getExePath()
  {
    char result[ 500 ];
    ssize_t count = readlink( "/proc/self/exe", result, 500 );
    return std::string( result, (count > 0) ? count : 0 );
  }

};

}
}
#endif //RAI_DATA_LOGGER_HPP
