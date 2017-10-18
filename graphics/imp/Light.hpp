//
// Created by jhwangbo on 17. 4. 28.
//

#ifndef PROJECT_LIGHT_HPP
#define PROJECT_LIGHT_HPP

#include "vector"
#include "Eigen/Core"
#include "mutex"
#include "iostream"
#include "rai/RAI_Vector.hpp"

namespace rai{
namespace Graphics{

class Light{

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  void getPosition(rai::Vector<float>& pos){
    mtx.lock();
    pos = position;
    mtx.unlock();
  }

  void getAmbient(rai::Vector<float>& amb){
    mtx.lock();
    amb = ambient;
    mtx.unlock();
  }

  void getDiffuse(rai::Vector<float>& diff){
    mtx.lock();
    diff = diffuse;
    mtx.unlock();
  }

  void getSpecular(rai::Vector<float>& spec){
    mtx.lock();
    spec = specular;
    mtx.unlock();
  }

  void setPosition(rai::Vector<float>& pos){
    mtx.lock();
    position = pos;
    mtx.unlock();
  }

  void setAmbient(rai::Vector<float>& amb){
    mtx.lock();
    ambient = amb;
    mtx.unlock();
  }

  void setDiffuse(rai::Vector<float>& diff){
    mtx.lock();
    diffuse = diff;
    mtx.unlock();
  }

  void setSpecular(rai::Vector<float>& spec){
    mtx.lock();
    specular = spec;
    mtx.unlock();
  }

 private:
  rai::Vector<float> position = {-100.0,0.0,10.0};
  rai::Vector<float> ambient = {0.5,0.5,0.5};
  rai::Vector<float> diffuse = {1,1,1};
  rai::Vector<float> specular = {0.7,0.7,0.7};
  std::mutex mtx;
};

}
}

#endif //PROJECT_LIGHT_HPP
