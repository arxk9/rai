//
// Created by jhwangbo on 17. 5. 1.
//

#ifndef RAI_SUPEROBJECT_HPP
#define RAI_SUPEROBJECT_HPP

#include "Object.hpp"
#include "../imp/shader.hpp"
#include "../imp/shader_basic.h"
#include "rai/common/TypeDef.hpp"
#include "rai/RAI_Vector.hpp"

namespace rai {
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

  void setTrans(rai::Vector<Transform>& trans);
  void getTrans(rai::Vector<Transform>& trans);
  void turnOnGhost(bool ghostOn);
  void drawSnapshot(Camera *camera,  Light *light, float transparency);

  bool visible = true;
  rai::Vector<Object*> objs;
  Shader* shader = nullptr;
  rai::Vector<rai::Vector<Transform> > ghosts;
  rai::Vector<Transform> currentPose;
  float ghostTransparency;
  int maxGhostN = 0;
  int oldestGhost = 0;

};

}
}
}

#endif //RAI_SUPEROBJECT_HPP
