//
// Created by joonho on 03.06.17.
//


#include <rai/common/math/RAI_math.hpp>
#include "Biped_ghost.hpp"
#include "rai/common/TypeDef.hpp"

namespace rai {
namespace Graphics {
namespace Obj {

Biped_ghost::Biped_ghost() :
    base(std::string(getenv("RAI_ROOT")) + "/RAI/taskModules/bipedLocomotion/cadModel/biped/Base_V1.dae", 0.001),

    haa_l(std::string(getenv("RAI_ROOT")) + "/RAI/taskModules/bipedLocomotion/cadModel/biped/Haa_V1.dae", 0.001),
    hfe_l(std::string(getenv("RAI_ROOT")) + "/RAI/taskModules/bipedLocomotion/cadModel/biped/Hfe_V1.dae", 0.001),
    thigh_l(std::string(getenv("RAI_ROOT")) + "/RAI/taskModules/bipedLocomotion/cadModel/biped/Thigh_V1.dae", 0.001),
    shank_l(std::string(getenv("RAI_ROOT")) + "/RAI/taskModules/bipedLocomotion/cadModel/biped/Shank_V1.dae", 0.001),
//    afe_l(std::string(getenv("RAI_ROOT")) + "/RAI/taskModules/bipedLocomotion/cadModel/biped/Afe_V1.dae", 0.001),

    haa_r(std::string(getenv("RAI_ROOT")) + "/RAI/taskModules/bipedLocomotion/cadModel/biped/Haa_V1.dae", 0.001),
    hfe_r(std::string(getenv("RAI_ROOT")) + "/RAI/taskModules/bipedLocomotion/cadModel/biped/Hfe_V1.dae", 0.001),
    thigh_r(std::string(getenv("RAI_ROOT")) + "/RAI/taskModules/bipedLocomotion/cadModel/biped/Thigh_V1.dae", 0.001),
    shank_r(std::string(getenv("RAI_ROOT")) + "/RAI/taskModules/bipedLocomotion/cadModel/biped/Shank_V1.dae", 0.001),
//    afe_r(std::string(getenv("RAI_ROOT")) + "/RAI/taskModules/bipedLocomotion/cadModel/biped/Afe_V1.dae", 0.001),
    foot_l(0.05),
    foot_r(0.05)
//    foot_l(std::string(getenv("RAI_ROOT")) + "/RAI/taskModules/bipedLocomotion/cadModel/biped/Foot_V1.dae", 0.001),
//    foot_r(std::string(getenv("RAI_ROOT")) + "/RAI/taskModules/bipedLocomotion/cadModel/biped/Foot_V1.dae", 0.001)
{

  std::vector<float> red = {1, 0, 0};
  std::vector<float> white = {1, 1, 1};
  std::vector<float> blue = {0, 0.5, 1};
  std::vector<float> gray = {0.7, 0.7, 0.7};

  base.setColor(white);
  haa_l.setColor(gray);

  hfe_l.setColor(gray);
  thigh_l.setColor(gray);
  shank_l.setColor(gray);

  haa_r.setColor(gray);
  hfe_r.setColor(gray);
  thigh_r.setColor(gray);
  shank_r.setColor(gray);



  foot_l.setColor({1, 0.1, 0.1});
  foot_r.setColor({0.1, 1, 0.1});

  float t = 0.5;
  float t2 = 0.5;
  base.setTransparency(t);
  haa_l.setTransparency(t);
  hfe_l.setTransparency(t);
  thigh_l.setTransparency(t);
  shank_l.setTransparency(t);
  haa_r.setTransparency(t);
  hfe_r.setTransparency(t2);
  thigh_r.setTransparency(t2);
  shank_r.setTransparency(t2);
  foot_l.setTransparency(t);
  foot_r.setTransparency(t2);


  objs.push_back(&base);

  objs.push_back(&haa_l);
  objs.push_back(&hfe_l);
  objs.push_back(&thigh_l);
  objs.push_back(&shank_l);
//  objs.push_back(&afe_l);
  objs.push_back(&foot_l);

  objs.push_back(&haa_r); ///
  objs.push_back(&hfe_r); ///
  objs.push_back(&thigh_r); ///
  objs.push_back(&shank_r); ///
//  objs.push_back(&afe_r);

//  objs.push_back(&foot_l);
//  objs.push_back(&foot_r);
  objs.push_back(&foot_r);

  defaultPose_.resize(11);
  for (auto &pose : defaultPose_)
    pose.setIdentity();

  /// manual adjustment
  ///base
  defaultPose_[0](0, 3) = -0.2;
  defaultPose_[4](2, 3) = 0.03;
  defaultPose_[9](2, 3) = 0.03;

  ///pelvis
  rai::Math::MathFunc::rotateHTabout_z_axis(defaultPose_[6], M_PI);
  rai::Math::MathFunc::rotateHTabout_y_axis(defaultPose_[6], M_PI);
  rai::Math::MathFunc::rotateHTabout_z_axis(defaultPose_[7], M_PI);
  ///leg
  rai::Math::MathFunc::rotateHTabout_z_axis(defaultPose_[8], M_PI);
  rai::Math::MathFunc::rotateHTabout_z_axis(defaultPose_[9], M_PI);
}

Biped_ghost::~Biped_ghost() {}

void Biped_ghost::init() {
  for (auto *body: objs)
    body->init();
  shader = new Shader_basic;
}

void Biped_ghost::destroy() {
  for (auto *body: objs)
    body->destroy();
  delete shader;
}

void Biped_ghost::setPose(std::vector<HomogeneousTransform> &bodyPose) {
  for (int i = 0; i < objs.size(); i++) {
    HomogeneousTransform ht = bodyPose[i] * defaultPose_[i];
    objs[i]->setPose(ht);
  }
}

}
}
}