#ifndef CAMERA_INCLUDED_H
#define CAMERA_INCLUDED_H

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <cmath>
#include <iostream>
#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <obj/Object.hpp>
#include "vector3d.h"

namespace rai {
namespace Graphics {
struct Camera {
 public:

  Camera(const glm::vec3 &pos, float fov, float aspect, float zNear, float zFar);
  void update();
  void GetVP(glm::mat4& vp);
  void GetPos(glm::vec3& position);
  void Control(SDL_Event e);
  void follow(rai::Graphics::Obj::Object* obj, Eigen::Vector3d pos);

 protected:
 private:

  rai::Graphics::Obj::Object* toFollowObj = nullptr;
  glm::vec4 relativePos;
  glm::vec4 rotationPitcAxis;
  glm::mat4 vp_;

  vector3d loc;
  float camPitch, camYaw;
  float camAngularSpeed;
  float camLinearSpeed = 0.1;

  bool mi = true;
  void lockCamera();
  bool mousePressedLastTimeStep = false;
  int prevMx = -1, prevMy = -1;
  std::mutex mtx;
  glm::mat4 projection;
  glm::vec3 pos;
  glm::vec3 forward;
  glm::vec3 up;
  const Uint8 *keyState;
  unsigned switchTime = 0;

};
}
}
#endif
