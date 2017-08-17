//
// Created by jhwangbo on 17. 4. 27.
//
#include "camera.h"

namespace RAI {
namespace Graphics {
void Camera::lockCamera() {
  if (camPitch > 90)
    camPitch = 90;
  if (camPitch < -90)
    camPitch = -90;
  if (camYaw < 0.0)
    camYaw += 360.0;
  if (camYaw > 360.0)
    camYaw -= 360;
}

Camera::Camera(const glm::vec3 &pos, float fov, float aspect, float zNear, float zFar) {
  this->pos = pos;
  this->forward = glm::vec3(0.0f, 0.0f, 1.0f);
  this->up = glm::vec3(0.0f, 0.0f, 1.0f);
  this->projection = glm::perspective(fov, aspect, zNear, zFar);

  camPitch = 0;
  camYaw = 0;
  camAngularSpeed = 0.2;
  camLinearSpeed = 0.1;
  keyState = SDL_GetKeyboardState(NULL);
}

void Camera::update() {
  mtx.lock();
  float sinYaw = sin(camYaw / 180 * M_PI);
  float cosYaw = cos(camYaw / 180 * M_PI);
  float sinPitch = sin(camPitch / 180 * M_PI);
  float cosPitch = cos(camPitch / 180 * M_PI);

  float lx = cosYaw * cosPitch;
  float ly = sinYaw * cosPitch;
  float lz = sinPitch;

  glm::vec3 offset;
  offset.x = lx * 10;
  offset.y = ly * 10;
  offset.z = lz * 10;

  if (mi)
    vp_ = projection * glm::lookAt(pos, pos + offset, up);
  else {
    Transform trans;
    toFollowObj->getTransform(trans);
    vp_ = projection * glm::lookAt(*trans.GetPos() + glm::vec3(relativePos), *trans.GetPos(), up);
  }

  mtx.unlock();
}

void Camera::GetVP(glm::mat4 &vp) {
  mtx.lock();
  vp = vp_;
  mtx.unlock();
}

void Camera::GetPos(glm::vec3 &position) {
  mtx.lock();
  position = pos;
  mtx.unlock();
}

void Camera::Control(SDL_Event e) {
  std::lock_guard<std::mutex> guad(mtx);
  if (mi) {
    float sinYaw = sin(camYaw / 180 * M_PI);
    float cosYaw = cos(camYaw / 180 * M_PI);
    float sinPitch = sin(camPitch / 180 * M_PI);
    float cosPitch = cos(camPitch / 180 * M_PI);

    float lx = cosYaw * cosPitch;
    float ly = sinYaw * cosPitch;
    float lz = sinPitch;

    int tmpx = 0, tmpy = 0;
    if (mousePressedLastTimeStep) {
      SDL_GetMouseState(&tmpx, &tmpy);
      camYaw += camAngularSpeed * (tmpx - prevMx);
      camPitch += camAngularSpeed * (tmpy - prevMy);
    }

    Uint32 mbuttonState = SDL_GetMouseState(&prevMx, &prevMy);

    if (mbuttonState == SDL_BUTTON_LEFT)
      mousePressedLastTimeStep = true;

    if (mbuttonState == 0)
      mousePressedLastTimeStep = false;

    lockCamera();
    glm::vec3 left, right;

    if (keyState[SDL_SCANCODE_W])
      pos += camLinearSpeed * glm::vec3(lx, ly, lz);

    if (keyState[SDL_SCANCODE_S])
      pos -= camLinearSpeed * glm::vec3(lx, ly, lz);

    if (keyState[SDL_SCANCODE_A]) {
      left = -camLinearSpeed * glm::normalize(glm::cross(glm::vec3(lx, ly, lz), up));
      pos += left;
    }

    if (keyState[SDL_SCANCODE_D]) {
      right = camLinearSpeed * glm::normalize(glm::cross(glm::vec3(lx, ly, lz), up));
      pos += right;
    }

    if (keyState[SDL_SCANCODE_KP_PLUS])
      camLinearSpeed *= 1.05;

    if (keyState[SDL_SCANCODE_KP_MINUS])
      camLinearSpeed /= 1.05;
  } else {

    if (switchTime > 3) {
      if (keyState[SDL_SCANCODE_LEFT]) {
        relativePos = glm::rotate(glm::radians(22.5f), glm::vec3(0, 0, 1)) * relativePos;
        rotationPitcAxis = glm::rotate(glm::radians(22.5f), glm::vec3(0, 0, 1)) * rotationPitcAxis;
        switchTime = 0;
      } else if (keyState[SDL_SCANCODE_RIGHT]) {
        relativePos = glm::rotate(glm::radians(-22.5f), glm::vec3(0, 0, 1)) * relativePos;
        rotationPitcAxis = glm::rotate(glm::radians(-22.5f), glm::vec3(0, 0, 1)) * rotationPitcAxis;
        switchTime = 0;
      } else if (keyState[SDL_SCANCODE_UP]) {
        relativePos = glm::rotate(glm::radians(-22.5f), glm::vec3(rotationPitcAxis)) * relativePos;
        switchTime = 0;
      } else if (keyState[SDL_SCANCODE_DOWN]) {
        relativePos = glm::rotate(glm::radians(-22.5f), glm::vec3(rotationPitcAxis)) * relativePos;
        switchTime = 0;
      }
    }
  }

  if (keyState[SDL_SCANCODE_SPACE] && switchTime > 10) {
    if (!toFollowObj) {
      std::cout << "specify which object to follow" << std::endl;
      return;
    }
    mi = !mi;
    camPitch = 0;
    camYaw = 0;
    switchTime = 0;
  }

  switchTime++;
}

void Camera::follow(RAI::Graphics::Obj::Object *obj, Eigen::Vector3d pos) {
  mtx.lock();
  toFollowObj = obj;
  relativePos = glm::vec4(pos[0], pos[1], pos[2], 0);
  rotationPitcAxis = glm::vec4(pos[1], pos[0], 0, 0);
  glm::normalize(rotationPitcAxis);
  mi = false;
  mtx.unlock();
}

}
}