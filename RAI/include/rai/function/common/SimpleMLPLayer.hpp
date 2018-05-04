//
// Created by jhwangbo on 17.04.17.
//

#ifndef RAI_SIMPLEMLP_HPP
#define RAI_SIMPLEMLP_HPP

#include <string>
#include <vector>
#include <Eigen/Dense>
#include <cstdlib>
#include "iostream"
#include <fstream>
#include <cmath>
#include <rai/noiseModel/NormalDistributionNoise.hpp>

namespace rai {

namespace FuncApprox {

template<typename Dtype, int StateDim, int ActionDim>
class MLP_fullyconnected {

 public:
  using Noise_ = Noise::NormalDistributionNoise<Dtype, ActionDim>;
  typedef Eigen::Matrix<Dtype, ActionDim, 1> Action;
  typedef Eigen::Matrix<Dtype, StateDim, 1> State;

  MLP_fullyconnected(std::string fileName, std::string activation, std::vector<int> hiddensizes) :
  cov(Eigen::Matrix<Dtype, ActionDim, ActionDim>::Identity()), act_(activation), noise_(cov) {
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

    layersizes.push_back(StateDim);
    layersizes.reserve(layersizes.size() + hiddensizes.size());
    layersizes.insert(layersizes.end(), hiddensizes.begin(), hiddensizes.end());
    layersizes.push_back(ActionDim);
    ///[input hidden output]

    if(activation == "tanh") isTanh=true;

    params.resize(2 * (layersizes.size() - 1));
    Ws.resize(layersizes.size() - 1);
    bs.resize(layersizes.size() - 1);
    Stdev.resize(ActionDim);

    std::stringstream parameterFileName;
    std::ifstream indata;
    indata.open(fileName);
    LOG_IF(FATAL, !indata) << "MLP file does not exists!" << std::endl;
    std::string line;
    getline(indata, line);
    std::stringstream lineStream(line);
    std::string cell;


    ///assign parameters
    for (int i = 0; i < params.size(); i++) {
      int paramSize = 0;

      if (i % 2 == 0) ///W resize
      {
        Ws[i / 2].resize(layersizes[i / 2 + 1], layersizes[i / 2]);
        params[i].resize(layersizes[i / 2] * layersizes[i / 2 + 1]);
      }
      if (i % 2 == 1) ///b resize
      {
        bs[(i - 1) / 2].resize(layersizes[(i + 1) / 2]);
        params[i].resize(layersizes[(i + 1) / 2]);
      }

      while (std::getline(lineStream, cell, ',')) { ///Read param
        params[i](paramSize++) = std::stod(cell);
        if (paramSize == params[i].size()) break;
      }
      if (i % 2 == 0) ///W copy
        memcpy(Ws[i / 2].data(), params[i].data(), sizeof(Dtype) * Ws[i / 2].size());
      if (i % 2 == 1) ///b copy
        memcpy(bs[(i - 1) / 2].data(), params[i].data(), sizeof(Dtype) * bs[(i - 1) / 2].size());
    }
    int cnt = 0;

    while (std::getline(lineStream, cell, ',')) {
      Stdev[cnt++] = std::stod(cell);
      if (cnt == ActionDim) break;
    }

    Action temp;
    temp = Stdev;
    temp = temp.array().square(); //var
    noise_.initializeNoise();
    noise_.updateCovariance(temp.asDiagonal());
  }

  Action forward(State &state) {
    Eigen::Matrix<Dtype,-1,1> temp= state;
    for (int cnt = 0; cnt < Ws.size() - 1; cnt++) {
      temp = Ws[cnt] * temp + bs[cnt];

      if (isTanh)
      temp = temp.array().tanh();
      else temp = temp.cwiseMax(0.0);
    }
    temp = Ws.back() * temp + bs.back(); /// output layer
    return temp;
  }




  Action noisify(Action actionMean) {
    return noise_.noisify(actionMean);
  }

 private:

//Eigen::MatrixXd output_;
  std::vector<Eigen::Matrix<Dtype,-1,1>> params;
  std::vector<Eigen::Matrix<Dtype,-1,-1>> Ws;
  std::vector<Eigen::Matrix<Dtype,-1,1>> bs;
  Action Stdev;

  std::vector<int> layersizes;
  std::string act_;
  Eigen::Matrix<Dtype, ActionDim, ActionDim> cov;
  Noise_ noise_;
  bool isTanh=false;

};

}

}

#endif //RAI_SIMPLEMLP_HPP
