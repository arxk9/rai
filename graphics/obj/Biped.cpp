//
// Created by joonho on 07.05.17.
//


#include <math/RAI_math.hpp>
#include "Biped.hpp"
#include "TypeDef.hpp"

namespace RAI {
namespace Graphics {
namespace Obj {

Biped::Biped():
    base(std::string(getenv("RAI_ROOT")) + "/RAI/taskModules/bipedLocomotion/cadModel/biped/Base_V1.dae", 0.001),

    haa_l(std::string(getenv("RAI_ROOT")) + "/RAI/taskModules/bipedLocomotion/cadModel/biped/Haa_V1.dae", 0.001),
    hfe_l(std::string(getenv("RAI_ROOT")) + "/RAI/taskModules/bipedLocomotion/cadModel/biped/Hfe_V1.dae", 0.001),
    thigh_l(std::string(getenv("RAI_ROOT")) + "/RAI/taskModules/bipedLocomotion/cadModel/biped/Thigh_V1.dae", 0.001),
    shank_l(std::string(getenv("RAI_ROOT")) + "/RAI/taskModules/bipedLocomotion/cadModel/biped/Shank_V1.dae", 0.001),
    afe_l(std::string(getenv("RAI_ROOT")) + "/RAI/taskModules/bipedLocomotion/cadModel/biped/Afe_V1.dae", 0.001),

    haa_r(std::string(getenv("RAI_ROOT")) + "/RAI/taskModules/bipedLocomotion/cadModel/biped/Haa_V1.dae", 0.001),
    hfe_r(std::string(getenv("RAI_ROOT")) + "/RAI/taskModules/bipedLocomotion/cadModel/biped/Hfe_V1.dae", 0.001),
    thigh_r(std::string(getenv("RAI_ROOT")) + "/RAI/taskModules/bipedLocomotion/cadModel/biped/Thigh_V1.dae", 0.001),
    shank_r(std::string(getenv("RAI_ROOT")) + "/RAI/taskModules/bipedLocomotion/cadModel/biped/Shank_V1.dae", 0.001),
    afe_r(std::string(getenv("RAI_ROOT")) + "/RAI/taskModules/bipedLocomotion/cadModel/biped/Afe_V1.dae", 0.001),
//    foot_l(0.05),
//    foot_r(0.05)
    foot_l(std::string(getenv("RAI_ROOT")) + "/RAI/taskModules/bipedLocomotion/cadModel/biped/Foot_V1.dae", 0.001),
    foot_r(std::string(getenv("RAI_ROOT")) + "/RAI/taskModules/bipedLocomotion/cadModel/biped/Foot_V1.dae", 0.001)
{
  std::vector<float> red = {1, 0, 0};
  std::vector<float> white = {1, 1, 1};
  std::vector<float> blue = {0, 0.5, 1};
  std::vector<float> gray = {0.7, 0.7, 0.7};
  base.setColor(white);
  haa_l.setColor(gray);
  hfe_l.setColor(blue);
  thigh_l.setColor(blue);
  shank_l.setColor(blue);
  haa_r.setColor(gray);
  hfe_r.setColor(blue);
  thigh_r.setColor(blue);
  shank_r.setColor(blue);
  afe_l.setColor(red);
  afe_r.setColor(red);
  foot_l.setColor(red);
  foot_r.setColor(red);



  objs.push_back(&base);

  objs.push_back(&haa_l);
  objs.push_back(&hfe_l);
  objs.push_back(&thigh_l);
  objs.push_back(&shank_l);
  objs.push_back(&afe_l);

  objs.push_back(&haa_r); ///
  objs.push_back(&hfe_r); ///
  objs.push_back(&thigh_r); ///
  objs.push_back(&shank_r); ///
  objs.push_back(&afe_r);

  objs.push_back(&foot_l);
  objs.push_back(&foot_r);


//  objs.push_back(&foot_l);
//  objs.push_back(&foot_r);
  defaultPose_.resize(13);
  for(auto& pose : defaultPose_)
    pose.setIdentity();

  /// manual adjustment
  ///base
  defaultPose_[0](0,3) = -0.2;
  ///pelvis
  RAI::Math::MathFunc::rotateHTabout_z_axis(defaultPose_[6], M_PI );
  RAI::Math::MathFunc::rotateHTabout_y_axis(defaultPose_[6], M_PI );
  RAI::Math::MathFunc::rotateHTabout_z_axis(defaultPose_[7], M_PI );
  ///leg
  RAI::Math::MathFunc::rotateHTabout_z_axis(defaultPose_[8], M_PI);
  RAI::Math::MathFunc::rotateHTabout_z_axis(defaultPose_[9], M_PI);
  ///ankle
  RAI::Math::MathFunc::rotateHTabout_x_axis(defaultPose_[10], M_PI );
}

Biped::~Biped(){}

void Biped::init(){
  for(auto* body: objs)
    body->init();
  shader = new Shader_basic;
}

void Biped::destroy(){
  for(auto* body: objs)
    body->destroy();
  delete shader;
}

void Biped::setPose(std::vector<HomogeneousTransform> &bodyPose) {
  for (int i = 0; i < objs.size(); i++) {
    HomogeneousTransform ht = bodyPose[i] * defaultPose_[i];
    objs[i]->setPose(ht);
  }
}

}
}
}