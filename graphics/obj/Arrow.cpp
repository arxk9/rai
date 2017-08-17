//
// Created by joonho on 19.05.17.
//
#include "Arrow.hpp"

namespace RAI {
namespace Graphics {
namespace Obj {

Arrow::Arrow(float r1, float r2, float l1, float l2) {
  int slices = 20;
  int position = 0;
//  bottom
  positions.push_back(glm::vec3(0, 0, 0));
  normals.push_back(glm::vec3(-1, 0, 0));

  for (int i = 0; i < slices; i++) {
    float theta = 2 * M_PI / slices * i;
    float y = r1 * std::cos(theta);
    float z = r1 * std::sin(theta);
    positions.push_back(glm::vec3(0, y, z));
    normals.push_back(glm::vec3(-r1, y, z));
  }
  for (int i = 1; i < slices + 1; i++) {
    indices.push_back(0);
    indices.push_back(i % slices + 1);
    indices.push_back(i);
  }
  position = slices;
//side
  for (int i = 0; i < slices; i++) {
    float theta = 2 * M_PI / slices * i;
    float y = r1 * std::cos(theta);
    float z = r1 * std::sin(theta);
    positions.push_back(glm::vec3(l1, y, z));
    normals.push_back(glm::vec3(0, y, z));
  }
  for (int i = 1; i < slices + 1; i++) {
    indices.push_back(i);
    indices.push_back(i % slices + 1);
    indices.push_back(position+ i);

    indices.push_back(position + i);
    indices.push_back(i % slices + 1);
    indices.push_back(position + i % slices + 1);
  }

//Head bottom
  positions.push_back(glm::vec3(l1, 0, 0));
  normals.push_back(glm::vec3(1 , 0, 0));
  position += slices + 1;

  for (int i = 0; i < slices; i++) {
    float theta = 2 * M_PI / slices * i;
    float y = r2 * std::cos(theta);
    float z = r2 * std::sin(theta);
    positions.push_back(glm::vec3(l1, y, z));
    normals.push_back(glm::vec3(l2, y, z));
  }
  for (int i = 1; i < slices + 1; i++) {
    indices.push_back(position);
    indices.push_back(i % slices + 1+ position);
    indices.push_back(i + position);
  }

  //Head
  positions.push_back(glm::vec3(l1 + l2, 0, 0));
  normals.push_back(glm::vec3(1 , 0, 0));

  for (int i = 1; i < slices + 1; i++) {
    indices.push_back(i +position);
    indices.push_back(i % slices + 1 +position);
    indices.push_back(position + slices + 1);
  }
}

}
}
}



