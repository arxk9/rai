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
#include "rai/RAI_Vector.hpp"

namespace rai {
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

  void setLightProp(rai::Vector<float>& amb, rai::Vector<float>& diff, rai::Vector<float>& spec, float shine);

  void setColor(rai::Vector<float> colorL);

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

  void getColor(rai::Vector<float>& clr);
  void getLightPropAmb(rai::Vector<float>& amb);
  void getLightPropDiff(rai::Vector<float>& diff);
  void getLightPropSpec(rai::Vector<float>& spec);
  void getShiness(float& shine);

 protected:
  void registerToGPU();
  Transform transform;
  Transform tempTransform;
  bool tempTransformOn = false;
  glm::mat4 scaleMat_;
  rai::Vector<float> color_ = {0.7, 0.7, 0.7};
  rai::Vector<float> amb_m = {0.3, 0.3, 0.3};
  rai::Vector<float> diff_m = {1.0,1.0,1.0};
  rai::Vector<float> spec_m = {1,1,1};
  float transparency_ = 1.0;
  float shininess = 100;
  rai::Vector<glm::vec3> colorsCoords;
  bool visible = true;
  rai::Vector<glm::vec3> positions;
  rai::Vector<glm::vec2> texCoords;
  rai::Vector<glm::vec3> normals;
  rai::Vector<unsigned int> indices;
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
