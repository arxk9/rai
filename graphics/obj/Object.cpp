//
// Created by jhwangbo on 17. 4. 29.
//
#include "Object.hpp"

namespace rai {
namespace Graphics {
namespace Obj {

void Object::draw() {
  std::lock_guard<std::mutex> guard(mtx);
  glBindVertexArray(m_vertexArrayObject);
  glDrawElementsBaseVertex(GL_TRIANGLES, m_numIndices, GL_UNSIGNED_INT, 0, 0);
  glBindVertexArray(0);
}

void Object::init() {
  std::lock_guard<std::mutex> guard(mtx);
  registerToGPU();
}

void Object::destroy() {
  glDeleteBuffers(NUM_BUFFERS, m_vertexArrayBuffers);
  glDeleteVertexArrays(1, &m_vertexArrayObject);
}

void Object::setPose(Eigen::Vector3d &position, Eigen::Matrix3d &rotationMat) {
  std::lock_guard<std::mutex> guard(mtx);
  Eigen::Quaternion<double> quat(rotationMat);
  glm::quat quatglm = glm::quat(quat.w(), quat.x(), quat.y(), quat.z());
  glm::vec3 pos(position(0), position(1), position(2));
  transform.SetRot(quatglm);
  transform.SetPos(pos);
}

void Object::setPose(Eigen::Vector3d &position, Eigen::Vector4d &quat) {
  std::lock_guard<std::mutex> guard(mtx);
  glm::quat quatglm = glm::quat(quat(0), quat(1), quat(2), quat(3));
  glm::vec3 pos(position(0), position(1), position(2));
  transform.SetRot(quatglm);
  transform.SetPos(pos);
}

void Object::setPos(Eigen::Vector3d &position) {
  std::lock_guard<std::mutex> guard(mtx);
  glm::vec3 pos(position(0), position(1), position(2));
  transform.SetPos(pos);
}

void Object::setOri(Eigen::Vector4d &quat) {
  std::lock_guard<std::mutex> guard(mtx);
  glm::quat quatglm = glm::quat(quat(0), quat(1), quat(2), quat(3));
  transform.SetRot(quatglm);
}

void Object::setOri(Eigen::Matrix3d &rotationMat) {
  std::lock_guard<std::mutex> guard(mtx);
  Eigen::Quaternion<double> quat(rotationMat);
  glm::quat quatglm = glm::quat(quat.w(), quat.x(), quat.y(), quat.z());
  transform.SetRot(quatglm);
}

void Object::setPose(Eigen::Matrix4d &ht) {
  Eigen::Vector3d pos = ht.topRightCorner(3, 1);
  Eigen::Matrix3d rot = ht.topLeftCorner(3, 3);
  setPose(pos, rot);
}

void Object::setTransform(Transform& trans) {
  transform = trans;
}

void Object::setLightProp(rai::Vector<float> &amb, rai::Vector<float> &diff, rai::Vector<float> &spec, float shine) {
  std::lock_guard<std::mutex> guard(mtx);
  amb_m = amb;
  diff_m = diff;
  spec_m = spec;
  shininess = shine;
}

void Object::setColor(rai::Vector<float> colorL) {
  std::lock_guard<std::mutex> guard(mtx);
  color_ = colorL;
}

void Object::setTransparency(float transparency) {
  std::lock_guard<std::mutex> guard(mtx);
  transparency_ = transparency;
}

float Object::getTransparency() {
  std::lock_guard<std::mutex> guard(mtx);
  return transparency_;
}

void Object::setScale(double scale) {
  std::lock_guard<std::mutex> guard(mtx);
  scaleMat_ = glm::scale(glm::vec3(scale, scale, scale));
}

void Object::setScale(double scale1, double scale2, double scale3) {
  std::lock_guard<std::mutex> guard(mtx);
  scaleMat_ = glm::scale(glm::vec3(scale1, scale2, scale3));
}

glm::mat4 Object::getScale() {
  std::lock_guard<std::mutex> guard(mtx);
  return scaleMat_;
}

void Object::getTransform(Transform& trans) {
  std::lock_guard<std::mutex> guard(mtx);
  if(tempTransformOn)
    trans = tempTransform;
  else
    trans = transform;
}

void Object::setTempTransform(Transform& trans) {
  tempTransform = trans;
}

void Object::getColor(rai::Vector<float> &clr) {
  std::lock_guard<std::mutex> guard(mtx);
  clr = color_;
}

void Object::getLightPropAmb(rai::Vector<float> &amb) {
  std::lock_guard<std::mutex> guard(mtx);
  amb = amb_m;
}

void Object::getLightPropDiff(rai::Vector<float> &diff) {
  std::lock_guard<std::mutex> guard(mtx);
  diff = diff_m;
}

void Object::getLightPropSpec(rai::Vector<float> &spec) {
  std::lock_guard<std::mutex> guard(mtx);
  spec = spec_m;
}

void Object::getShiness(float& shine) {
  std::lock_guard<std::mutex> guard(mtx);
  shine = shininess;
}

void Object::registerToGPU() {
  m_numIndices = indices.size();

  glGenVertexArrays(1, &m_vertexArrayObject);
  glBindVertexArray(m_vertexArrayObject);

  glGenBuffers(NUM_BUFFERS, m_vertexArrayBuffers);

  glBindBuffer(GL_ARRAY_BUFFER, m_vertexArrayBuffers[POSITION_VB]);
  glBufferData(GL_ARRAY_BUFFER,
               sizeof(positions[0]) * positions.size(),
               &positions[0],
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

  glBindBuffer(GL_ARRAY_BUFFER, m_vertexArrayBuffers[TEXCOORD_VB]);
  glBufferData(GL_ARRAY_BUFFER,
               sizeof(texCoords[0]) * texCoords.size(),
               &texCoords[0],
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);

  glBindBuffer(GL_ARRAY_BUFFER, m_vertexArrayBuffers[NORMAL_VB]);
  glBufferData(GL_ARRAY_BUFFER, sizeof(normals[0]) * normals.size(), &normals[0], GL_STATIC_DRAW);
  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, 0);

  glBindBuffer(GL_ARRAY_BUFFER, m_vertexArrayBuffers[COLOR_VB]);
  glBufferData(GL_ARRAY_BUFFER, sizeof(colorsCoords[0]) * colorsCoords.size(), &colorsCoords[0], GL_STATIC_DRAW);
  glEnableVertexAttribArray(3);
  glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, 0);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_vertexArrayBuffers[INDEX_VB]);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER,
               sizeof(indices[0]) * indices.size(),
               &indices[0],
               GL_STATIC_DRAW);

  glBindVertexArray(0);
}

}
}
}
