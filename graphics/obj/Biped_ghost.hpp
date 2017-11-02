//
// Created by joonho on 03.06.17.
//


#include <rai/common/TypeDef.hpp>
#include "SuperObject.hpp"
#include "Mesh.hpp"
#include "Sphere.hpp"


namespace rai {
namespace Graphics {
namespace Obj {

class Biped_ghost : public SuperObject {

 public:

  Biped_ghost();
  ~Biped_ghost();
  void init();
  void destroy();
  void setPose(std::vector<HomogeneousTransform> &bodyPose);
  rai::Graphics::Obj::Object* basePtr(){ return &base; }
  rai::Graphics::Obj::Object* footlPtr(){ return &foot_l; }
  rai::Graphics::Obj::Object* footrPtr(){ return &foot_r; }


 private:

  rai::Graphics::Obj::Mesh base;
  rai::Graphics::Obj::Mesh haa_l;
  rai::Graphics::Obj::Mesh hfe_l;
  rai::Graphics::Obj::Mesh thigh_l;
  rai::Graphics::Obj::Mesh shank_l;
//  rai::Graphics::Obj::Mesh afe_l;

  rai::Graphics::Obj::Mesh haa_r;
  rai::Graphics::Obj::Mesh hfe_r;
  rai::Graphics::Obj::Mesh thigh_r;
  rai::Graphics::Obj::Mesh shank_r;
//  rai::Graphics::Obj::Mesh afe_r;

  rai::Graphics::Obj::Sphere foot_l;
  rai::Graphics::Obj::Sphere foot_r;

  std::vector<HomogeneousTransform> defaultPose_;
};

}
}
}