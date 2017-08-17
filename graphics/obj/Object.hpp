//
// Created by jhwangbo on 17. 4. 28.
//

#ifndef PROJECT_OBJECT_HPP
#define PROJECT_OBJECT_HPP
#include "Eigen/Geometry"
#include "../imp/transform.h"
#include "../imp/Light.hpp"
#include <mutex>

#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>

namespace RAI {
namespace Graphics{
namespace Obj {

enum MeshBufferPositions {
  POSITION_VB,
  TEXCOORD_VB,
  NORMAL_VB,
  COLOR_VB,
  INDEX_VB
};

class Object {
 public:

  virtual void draw();

  virtual void init();

  virtual void destroy();

  void setPose(Eigen::Vector3d &position, Eigen::Matrix3d &rotationMat);

  void setPose(Eigen::Vector3d &position, Eigen::Vector4d &quat);

  void setPos(Eigen::Vector3d &position);

  void setOri(Eigen::Vector4d &quat);

  void setOri(Eigen::Matrix3d &rotationMat);

  void setPose(Eigen::Matrix4d &ht);

  void setTransform(Transform& trans);

  void setLightProp(std::vector<float>& amb, std::vector<float>& diff, std::vector<float>& spec, float shine);

  void setColor(std::vector<float> colorL);

  void setTransparency(float transparency);

  float getTransparency();

  void setVisibility(bool visibility) {visible = visibility;}

  void setScale(double scale);

  void setScale(double scale1,double scale2,double scale3);

  glm::mat4 getScale();

  bool isVisible() {return visible;}

  void getTransform(Transform& trans);

  void setTempTransform(Transform& trans);

  void usingTempTransform(bool utt) {tempTransformOn = utt;};

  void getColor(std::vector<float>& clr);
  void getLightPropAmb(std::vector<float>& amb);
  void getLightPropDiff(std::vector<float>& diff);
  void getLightPropSpec(std::vector<float>& spec);
  void getShiness(float& shine);

 protected:
  void registerToGPU();
  Transform transform;
  Transform tempTransform;
  bool tempTransformOn = false;
  glm::mat4 scaleMat_;
  std::vector<float> color_ = {0.7, 0.7, 0.7};
  std::vector<float> amb_m = {0.3, 0.3, 0.3};
  std::vector<float> diff_m = {1.0,1.0,1.0};
  std::vector<float> spec_m = {1,1,1};
  float transparency_ = 1.0;
  float shininess = 100;
  std::vector<glm::vec3> colorsCoords;
  bool visible = true;
  std::vector<glm::vec3> positions;
  std::vector<glm::vec2> texCoords;
  std::vector<glm::vec3> normals;
  std::vector<unsigned int> indices;
  static const unsigned int NUM_BUFFERS = 5;
  GLuint m_vertexArrayObject;
  GLuint m_vertexArrayBuffers[NUM_BUFFERS];
  unsigned int m_numIndices;
  std::mutex mtx;
};

}
}
}

#endif //PROJECT_OBJECT_HPP
