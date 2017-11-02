//
// Created by joonho on 07.05.17.
//

#ifndef RAI_BIPED_HPP
#define RAI_BIPED_HPP
#include <rai/common/TypeDef.hpp>
#include "SuperObject.hpp"
#include "Mesh.hpp"
#include "Sphere.hpp"


namespace rai {
namespace Graphics {
namespace Obj {

class Biped : public SuperObject {

 public:

  Biped();
  ~Biped();
  void init();
  void destroy();
  void setPose(std::vector<HomogeneousTransform> &bodyPose);
  rai::Graphics::Obj::Object* basePtr(){ return &base; }

 private:


  rai::Graphics::Obj::Mesh base;
  rai::Graphics::Obj::Mesh haa_l;
  rai::Graphics::Obj::Mesh hfe_l;
  rai::Graphics::Obj::Mesh thigh_l;
  rai::Graphics::Obj::Mesh shank_l;
  rai::Graphics::Obj::Mesh afe_l;

  rai::Graphics::Obj::Mesh haa_r;
  rai::Graphics::Obj::Mesh hfe_r;
  rai::Graphics::Obj::Mesh thigh_r;
  rai::Graphics::Obj::Mesh shank_r;
  rai::Graphics::Obj::Mesh afe_r;

  rai::Graphics::Obj::Mesh foot_l;
  rai::Graphics::Obj::Mesh foot_r;

  std::vector<HomogeneousTransform> defaultPose_;
};

}
}
}
#endif //RAI_BIPED_HPP
