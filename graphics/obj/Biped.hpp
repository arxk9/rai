//
// Created by joonho on 07.05.17.
//

#ifndef RAI_BIPED_HPP
#define RAI_BIPED_HPP
#include <TypeDef.hpp>
#include "SuperObject.hpp"
#include "Mesh.hpp"
#include "Sphere.hpp"

namespace RAI {
namespace Graphics {
namespace Obj {

class Biped : public SuperObject {

 public:

  Biped();
  ~Biped();
  void init();
  void destroy();
  void setPose(std::vector<HomogeneousTransform> &bodyPose);
  RAI::Graphics::Obj::Object* basePtr(){ return &base; }

 private:


  RAI::Graphics::Obj::Mesh base;
  RAI::Graphics::Obj::Mesh haa_l;
  RAI::Graphics::Obj::Mesh hfe_l;
  RAI::Graphics::Obj::Mesh thigh_l;
  RAI::Graphics::Obj::Mesh shank_l;
  RAI::Graphics::Obj::Mesh afe_l;

  RAI::Graphics::Obj::Mesh haa_r;
  RAI::Graphics::Obj::Mesh hfe_r;
  RAI::Graphics::Obj::Mesh thigh_r;
  RAI::Graphics::Obj::Mesh shank_r;
  RAI::Graphics::Obj::Mesh afe_r;

  RAI::Graphics::Obj::Mesh foot_l;
  RAI::Graphics::Obj::Mesh foot_r;

  std::vector<HomogeneousTransform> defaultPose_;
};

}
}
}
#endif //RAI_BIPED_HPP
