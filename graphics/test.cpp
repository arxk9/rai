//
// Created by jhwangbo on 17. 4. 28.
//

#include "RAI_graphics.hpp"
#include "obj/Mesh.hpp"
#include "obj/Sphere.hpp"
#include "obj/Background.hpp"
#include "rai/RAI_core"

using namespace RAI::Graphics;


int main() {

  RAI_init();
  RAI_graphics graphics(800, 600);

  Obj::Mesh anymalBase(std::string(getenv("RAI_ROOT"))+"/graphics/res/anymal_base_1_2.dae", 0.001);
  anymalBase.setScale(5.0);
  anymalBase.setTransparency(0.3);

  Obj::Mesh terrain(std::string(getenv("RAI_ROOT"))+"/graphics/res/roughterrain.obj");
  Obj::Sphere sphere(1);
  Obj::Background background("sky");

  terrain.setColor({0.7f,0.2f,0.2f});

  Eigen::Vector3d pos; pos << 6, 0, 5;
  sphere.setPos(pos);
  graphics.addBackground(&background);
  graphics.addObject(&terrain);
  graphics.addObject(&anymalBase);
  graphics.addObject(&sphere);

  RAI_graphics::LightProp lprop;
  RAI_graphics::CameraProp cprop;
  cprop.toFollow = &anymalBase;
  Eigen::Vector3d relPos; relPos << 3, 0, 0.1;
  cprop.relativeDist = relPos;

  graphics.setCameraProp(cprop);
  graphics.setLightProp(lprop);

  graphics.start();
  mkdir((RAI_LOG_PATH+"/testVideo").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
//  graphics.savingSnapshots(RAI_LOG_PATH+"/testVideo", "testVideo");
  usleep(15e6);
//  graphics.images2Video();
  graphics.end();

  return 0;
}