
#include "rai/tasks/poleBalancing/visualizer/Pole_Visualizer.hpp"

namespace rai {
namespace Vis {

Pole_Visualizer::Pole_Visualizer() :
    graphics(600, 450),
    Pole(0.05, 1),
    Dot(0.055),
    Dot2(0.05),
    arrow(std::string(getenv("RAI_ROOT")) + "/RAI/taskModules/poleBalancing/cadModel/rotation.stl", 0.1) {

  Dot.setColor({0.3, 0.3, 0.3});
  Dot2.setColor({0.3, 0.3, 0.3});
  Pole.setColor({0.3, 0.3, 0.3});
  arrow.setColor({0, 0, 1});
  arrow.setTransparency(0.5);

  defaultPose_.setIdentity();
  rai::Math::MathFunc::rotateHTabout_z_axis(defaultPose_, M_PI_2);
  defaultPose_(0, 3) = +1.53 / 4 * 0.1;

  graphics.addObject(&Pole);
  graphics.addObject(&Dot);
  graphics.addObject(&Dot2);
  graphics.addObject(&arrow);

  graphics.setBackgroundColor(1, 1, 1, 1);

  Eigen::Vector3d relPos;
  relPos << -3, 0, 0;
  std::vector<float> pos = {-100, 0, 0}, spec = {0.7, 0.7, 0.7}, amb = {0.7, 0.7, 0.7}, diff = {0.7, 0.7, 0.7};

  Graphics::RAI_graphics::LightProp lprop;
  lprop.amb_light = amb;
  lprop.spec_light = spec;
  lprop.diff_light = diff;
  lprop.pos_light = pos;
  Graphics::RAI_graphics::CameraProp cprop;
  cprop.toFollow = &Pole;
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
  Eigen::Vector3d end;
  HomogeneousTransform arrowpose;
  RotationMatrix rotmat;
  arrowpose.setIdentity();
  end << 0, 0, 1;
  rotmat = bodyPose.topLeftCorner(3, 3);
  pos = bodyPose.topRightCorner(3, 1);

  arrowpose.topRightCorner(3, 1) = pos;

  end = rotmat * end;

  if (action < 0) {
    rai::Math::MathFunc::rotateHTabout_z_axis(arrowpose, M_PI);
    action = -action;
  }
  arrowpose;
  HomogeneousTransform ht = arrowpose * defaultPose_;

  Dot.setPos(pos);
  Pole.setPose(bodyPose);
  Dot2.setPos(end);

  arrow.setPose(ht);
  arrow.setScale(action);

}

Graphics::RAI_graphics *Pole_Visualizer::getGraphics() {
  return &graphics;
}
}
}