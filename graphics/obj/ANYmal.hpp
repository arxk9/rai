//
// Created by jhwangbo on 17. 5. 3.
//

#ifndef RAI_ANYMAL_HPP
#define RAI_ANYMAL_HPP

#include <rai/common/TypeDef.hpp>
#include "SuperObject.hpp"
#include "Mesh.hpp"

#include "Sphere.hpp"

namespace rai {
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
  rai::Graphics::Obj::Object* basePtr(){ return &base; }

 private:

  rai::Graphics::Obj::Mesh base;
  rai::Graphics::Obj::Mesh hip_lf;
  rai::Graphics::Obj::Mesh hip_rf;
  rai::Graphics::Obj::Mesh hip_lh;
  rai::Graphics::Obj::Mesh hip_rh;

  rai::Graphics::Obj::Mesh thigh_lf;
  rai::Graphics::Obj::Mesh thigh_rf;
  rai::Graphics::Obj::Mesh thigh_lh;
  rai::Graphics::Obj::Mesh thigh_rh;

  rai::Graphics::Obj::Mesh shank_lf;
  rai::Graphics::Obj::Mesh shank_rf;
  rai::Graphics::Obj::Mesh shank_lh;
  rai::Graphics::Obj::Mesh shank_rh;

//  rai::Graphics::Obj::Mesh foot_lf;
//  rai::Graphics::Obj::Mesh foot_rf;
//  rai::Graphics::Obj::Mesh foot_lh;
//  rai::Graphics::Obj::Mesh foot_rh;

  rai::Graphics::Obj::Sphere foot_lf;
  rai::Graphics::Obj::Sphere foot_rf;
  rai::Graphics::Obj::Sphere foot_lh;
  rai::Graphics::Obj::Sphere foot_rh;
  std::vector<HomogeneousTransform> defaultPose_;
};

}
}
}


#endif //RAI_ANYMAL_HPP
