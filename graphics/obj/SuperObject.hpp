//
// Created by jhwangbo on 17. 5. 1.
//

#ifndef RAI_SUPEROBJECT_HPP
#define RAI_SUPEROBJECT_HPP

#include "Object.hpp"
#include "../imp/shader.hpp"
#include "../imp/shader_basic.h"
#include "TypeDef.hpp"

namespace RAI {
namespace Graphics {
namespace Obj {

class SuperObject{

 public:

  virtual void init() = 0;
  virtual void destroy() = 0;

  void draw(Camera *camera,  Light *light);
  void setVisibility(bool visibility) {visible = visibility;}
  bool isVisible() {return visible;}
  void showGhosts(int maxGhosts, float transparency);
  void addGhostsNow();

 protected:

  void setTrans(std::vector<Transform>& trans);
  void getTrans(std::vector<Transform>& trans);
  void turnOnGhost(bool ghostOn);
  void drawSnapshot(Camera *camera,  Light *light, float transparency);

  bool visible = true;
  std::vector<Object*> objs;
  Shader* shader = nullptr;
  std::vector<std::vector<Transform> > ghosts;
  std::vector<Transform> currentPose;
  float ghostTransparency;
  int maxGhostN = 0;
  int oldestGhost = 0;

};

}
}
}

#endif //RAI_SUPEROBJECT_HPP
