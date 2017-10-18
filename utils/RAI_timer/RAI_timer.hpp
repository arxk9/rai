//
// Created by jhwangbo on 11.11.16.
// this method is relatively accurate down to 1 micro second.
//

#ifndef RAI_RAI_TIMER_HPP
#define RAI_RAI_TIMER_HPP

#define RAI_TIMER_WALL_CLOCK

#include <vector>
#include <float.h>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include "glog/logging.h"
#include <fstream>
#include <cstdlib>
#include <iostream>
#include "RAI_timer_items.hpp"

#ifdef RAI_TIMER_WALL_CLOCK
#include <sys/time.h>
#else
#include <time.h>
#define RAI_CLOCK_TYPE CLOCK_THREAD_CPUTIME_ID
#endif

namespace rai {
namespace Utils {

class Timer {
 public:
  Timer() : items_("Computation", -1) {
#ifdef RAI_TIMER_WALL_CLOCK
    gettimeofday(&timevalNow, nullptr);
    processStartTime_ = timevalNow.tv_sec + 1e-6 * timevalNow.tv_usec;
#else
    clock_gettime(RAI_CLOCK_TYPE, &timespec);
    processStartTime_ = timespec.tv_sec + 1e-9 * timespec.tv_nsec;
#endif
  }

  ~Timer() {
    std::ostringstream logPath;
    logPath << log_path_ << "/timerLog.rlog";
    std::ofstream logFile;
    logFile.open(logPath.str().c_str());
    logFile << "logging in seconds" << "\n";
    for (int idx = 0; idx < name_.size(); idx++)
      logFile << std::setw(30) << std::left << name_[idx] << "|" << "  N of samples: " << std::setw(8) << std::left
              << frameCount[idx] << "  " << "min: " << std::setw(15) << std::left << timeMin[idx] << "max: "
              << std::setw(15) << std::left << timeMax[idx] << "Avg: " << std::setw(15) << std::left << Avg[idx]
              << "Total: " << std::setw(15) << std::left << timeSum[idx] << "\n";
    logFile.close();
  }

  void startTimer(std::string name) {
    if(disableTimer) return;
    unsigned int idx = std::find(name_.begin(), name_.end(), name) - name_.begin();
    if (idx == name_.size()) {
      name_.push_back(name);
      timeMin.push_back(FLT_MAX);
      timeMax.push_back(0.0);
      timeSum.push_back(0.0);
      frameCount.push_back(0);
      Avg.push_back(0.0f);
      curTime.push_back(0.0);
      curTime_ns.push_back(0.0);
      allParent.clear();

      for (int itemID = 0; itemID < name_.size(); itemID++)
        if (curTime[itemID] != 0)
          allParent.push_back(itemID);

      items_.add_item(name, idx, allParent);
    }

    LOG_IF(WARNING, curTime[idx] != 0.0) << name <<": start timer is called twice consecutively. Call stopTimer first";
#ifdef RAI_TIMER_WALL_CLOCK
    gettimeofday(&timevalNow, nullptr);
    curTime[idx] = timevalNow.tv_sec + 1e-6 * timevalNow.tv_usec;
#else
    clock_gettime(RAI_CLOCK_TYPE, &timespec);
    curTime[idx] = timespec.tv_sec;
    curTime_ns[idx] = timespec.tv_nsec;
#endif
  }

  void stopTimer(std::string name) {
    if(disableTimer) return;
#ifdef RAI_TIMER_WALL_CLOCK
    gettimeofday(&timevalNow, nullptr);
#else
    clock_gettime(RAI_CLOCK_TYPE, &timespec);
#endif
    unsigned int idx = std::find(name_.begin(), name_.end(), name) - name_.begin();
    if (idx == name_.size()) {
      LOG(WARNING) << "stop timer is called before start timer";
      return;
    } else {
#ifdef RAI_TIMER_WALL_CLOCK
      timeLaps = timevalNow.tv_sec + 1e-6 * timevalNow.tv_usec - curTime[idx];
#else
      timeLaps = timespec.tv_sec - curTime[idx] + 1e-9 * (timespec.tv_nsec - curTime_ns[idx]);
#endif
      curTime[idx] = 0.0;
    }

    timeSum[idx] += timeLaps;
    timeMin[idx] = std::min(timeMin[idx], timeLaps);
    timeMax[idx] = std::max(timeMax[idx], timeLaps);
    frameCount[idx]++;
    Avg[idx] = timeSum[idx] / (float) frameCount[idx];
  }

  std::string getCurrentDataAndTime() {
    std::ostringstream dataAndTime;
    time_t t = time(0);   // get time now
    struct tm *now = localtime(&t);
    dataAndTime << (now->tm_year + 1900) << '-' << std::setfill('0') << std::setw(2) << (now->tm_mon + 1) << '-'
                << std::setw(2) << now->tm_mday << '-' << std::setw(2) << now->tm_hour << '-' << std::setw(2)
                << now->tm_min << '-' << std::setw(2) << now->tm_sec;
    return dataAndTime.str().c_str();
  }

  rai::Vector<double> &getTotalTime() {
    return timeSum;
  }

  rai::Vector<std::string> &getNames() {
    return name_;
  }


  Timer_items& getTimedItems(){
    for(int i=0; i< name_.size(); i++)
      items_.update_time_sum_(i, timeSum[i]);

    double totalTime = 0;
    for(int i=0; i< items_.subItems_.size(); i++)
      totalTime += items_.subItems_[i].time_sum_;

    items_.update_time_sum_(-1, totalTime);

    return items_;
  }

  void setLogPath(std::string& path){
    log_path_ = path;
  }

  void disable() {
    disableTimer = true;
  }

  void enable() {
    disableTimer = false;
  }

  double getGlobalElapsedTimeInSec(){
    double currentTime;
#ifdef RAI_TIMER_WALL_CLOCK
    gettimeofday(&timevalNow, nullptr);
    currentTime = timevalNow.tv_sec + 1e-6 * timevalNow.tv_usec;
#else
    clock_gettime(RAI_CLOCK_TYPE, &timespec);
    currentTime = timespec.tv_sec + 1e-9 * timespec.tv_nsec;
#endif
    return currentTime - processStartTime_;
  }

  double getGlobalElapsedTimeInMin(){
    double currentTime;
#ifdef RAI_TIMER_WALL_CLOCK
    gettimeofday(&timevalNow, nullptr);
    currentTime = timevalNow.tv_sec + 1e-6 * timevalNow.tv_usec;
#else
    clock_gettime(RAI_CLOCK_TYPE, &timespec);
    currentTime = timespec.tv_sec + 1e-9 * timespec.tv_nsec;
#endif
    return (currentTime - processStartTime_)/60.0;
  }

  double getGlobalElapsedTimeInHr(){
    double currentTime;
#ifdef RAI_TIMER_WALL_CLOCK
    gettimeofday(&timevalNow, nullptr);
    currentTime = timevalNow.tv_sec + 1e-6 * timevalNow.tv_usec;
#else
    clock_gettime(RAI_CLOCK_TYPE, &timespec);
    currentTime = timespec.tv_sec + 1e-9 * timespec.tv_nsec;
#endif
    return (currentTime - processStartTime_)/3600.0;
  }


 private:
  rai::Vector<std::string> name_;
  timeval timevalNow;
  struct timespec timespec;
  rai::Vector<double> timeSum;
  rai::Vector<double> timeMin;
  rai::Vector<double> timeMax;
  rai::Vector<int> frameCount;
  rai::Vector<int> allParent;
  rai::Vector<double> Avg;
  rai::Vector<double> curTime;
  rai::Vector<double> curTime_ns;
  double timeLaps;
  std::string log_path_;
  Timer_items items_;
  bool disableTimer = false;
  double processStartTime_;

};

}
}

#endif //RAI_RAI_TIMER_HPP
