//
// Created by joonho on 03.06.17.
//


#include <TypeDef.hpp>
#include "SuperObject.hpp"
#include "Mesh.hpp"
#include "Sphere.hpp"

namespace RAI {
namespace Graphics {
namespace Obj {

class Biped_ghost : public SuperObject {

 public:

  Biped_ghost();
  ~Biped_ghost();
  void init();
  void destroy();
  void setPose(std::vector<HomogeneousTransform> &bodyPose);
  RAI::Graphics::Obj::Object* basePtr(){ return &base; }
  RAI::Graphics::Obj::Object* footlPtr(){ return &foot_l; }
  RAI::Graphics::Obj::Object* footrPtr(){ return &foot_r; }


 private:

  RAI::Graphics::Obj::Mesh base;
  RAI::Graphics::Obj::Mesh haa_l;
  RAI::Graphics::Obj::Mesh hfe_l;
  RAI::Graphics::Obj::Mesh thigh_l;
  RAI::Graphics::Obj::Mesh shank_l;
//  RAI::Graphics::Obj::Mesh afe_l;

  RAI::Graphics::Obj::Mesh haa_r;
  RAI::Graphics::Obj::Mesh hfe_r;
  RAI::Graphics::Obj::Mesh thigh_r;
  RAI::Graphics::Obj::Mesh shank_r;
//  RAI::Graphics::Obj::Mesh afe_r;

  RAI::Graphics::Obj::Sphere foot_l;
  RAI::Graphics::Obj::Sphere foot_r;

  std::vector<HomogeneousTransform> defaultPose_;
};

}
}
}