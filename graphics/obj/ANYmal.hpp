//
// Created by jhwangbo on 17. 5. 3.
//

#ifndef RAI_ANYMAL_HPP
#define RAI_ANYMAL_HPP

#include <TypeDef.hpp>
#include "SuperObject.hpp"
#include "Mesh.hpp"

namespace RAI {
namespace Graphics {
namespace Obj {

class ANYmal : public SuperObject {

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ANYmal();
  ~ANYmal();
  void init();
  void destroy();
  void setPose(std::vector<HomogeneousTransform> &bodyPose);
  RAI::Graphics::Obj::Object* basePtr(){ return &base; }

 private:

  RAI::Graphics::Obj::Mesh base;
  RAI::Graphics::Obj::Mesh hip_lf;
  RAI::Graphics::Obj::Mesh hip_rf;
  RAI::Graphics::Obj::Mesh hip_lh;
  RAI::Graphics::Obj::Mesh hip_rh;

  RAI::Graphics::Obj::Mesh thigh_lf;
  RAI::Graphics::Obj::Mesh thigh_rf;
  RAI::Graphics::Obj::Mesh thigh_lh;
  RAI::Graphics::Obj::Mesh thigh_rh;

  RAI::Graphics::Obj::Mesh shank_lf;
  RAI::Graphics::Obj::Mesh shank_rf;
  RAI::Graphics::Obj::Mesh shank_lh;
  RAI::Graphics::Obj::Mesh shank_rh;

  RAI::Graphics::Obj::Mesh foot_lf;
  RAI::Graphics::Obj::Mesh foot_rf;
  RAI::Graphics::Obj::Mesh foot_lh;
  RAI::Graphics::Obj::Mesh foot_rh;
  std::vector<HomogeneousTransform> defaultPose_;
};

}
}
}


#endif //RAI_ANYMAL_HPP
