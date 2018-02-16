
#include "rai/tasks/poleBalancing/visualizer/Pole_Visualizer.hpp"

namespace rai {
namespace Vis {

Pole_Visualizer::Pole_Visualizer() :
    graphics(600, 450),
    Pole(0.05, 1),
    Dot(0.07),
    origin(0.1),
    arrow(std::string(getenv("RAI_ROOT")) + "/RAI/taskModules/poleBalancing/cadModel/rotation.stl", 0.1) {

  Dot.setColor({0.3, 0.9, 0.3});
  Pole.setColor({0.3, 0.3, 0.3});
  origin.setColor({0.8, 0.2, 0.2});
  arrow.setColor({0, 0, 1});
  arrow.setTransparency(0.5);
  origin.setPos({0,0,0});

  defaultPose_.setIdentity();
  rai::Math::MathFunc::rotateHTabout_z_axis(defaultPose_, M_PI_2);
  defaultPose_(0, 3) = +1.53 / 4 * 0.1;

  graphics.addObject(&Pole);
  graphics.addObject(&Dot);
  graphics.addObject(&arrow);
  graphics.addObject(&origin);

  graphics.setBackgroundColor(1, 1, 1, 1);

  Eigen::Vector3d relPos;
  relPos << -3, 0, 0;
  std::vector<float> pos = {-100, 0, 0}, spec = {0.7, 0.7, 0.7}, amb = {0.7, 0.7, 0.7}, diff = {0.7, 0.7, 0.7};

  rai_graphics::LightProp lprop;
  lprop.amb_light = amb;
  lprop.spec_light = spec;
  lprop.diff_light = diff;
  lprop.pos_light = pos;
  rai_graphics::CameraProp cprop;
  cprop.toFollow = &origin;
  cprop.relativeDist = relPos;

  graphics.setCameraProp(cprop);
  graphics.setLightProp(lprop);
  graphics.start();
}

Pole_Visualizer::~Pole_Visualizer() {
  graphics.end();
}

void Pole_Visualizer::drawWorld(HomogeneousTransform &bodyPose, double action) {
  Eigen::Vector3d pos;
  Eigen::Vector3d end, stick;
  HomogeneousTransform arrowpose, polePose;
  RotationMatrix rotmat;
  arrowpose.setIdentity();
  end << 0, 0, 1;
  stick << 0, 0, 0.5;

  rotmat = bodyPose.topLeftCorner(3, 3);
  pos = bodyPose.topRightCorner(3, 1);
  polePose.topLeftCorner(3,3) = rotmat;
  arrowpose.topRightCorner(3, 1) = pos;

  end = rotmat * end;
  stick = rotmat * stick;
  polePose.topRightCorner(3, 1) = stick;

  if (action < 0) {
    rai::Math::MathFunc::rotateHTabout_z_axis(arrowpose, M_PI);
    action = -action;
  }
  arrowpose;
  HomogeneousTransform ht = arrowpose * defaultPose_;

  Dot.setPos(end);
  Pole.setPose(polePose);

  arrow.setPose(ht);
  arrow.setScale(action);

}

rai_graphics::RAI_graphics *Pole_Visualizer::getGraphics() {
  return &graphics;
}
}
}