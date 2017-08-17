//
//
// Created by joonho on 19.05.17.
//

#include "Cylinder.hpp"

namespace RAI {
namespace Graphics {
namespace Obj {

Cylinder::Cylinder(float r, float l) {
  int slices = 20;
  int position = 0;
//  bottom
  positions.push_back(glm::vec3(0, 0, 0));
  normals.push_back(glm::vec3(0, 0, -1));

  for (int i = 0; i < slices; i++) {
    float theta = 2 * M_PI / slices * i;
    float y = r * std::cos(theta);
    float x = -r * std::sin(theta);
    positions.push_back(glm::vec3(x, y, 0));
//    normals.push_back(glm::vec3(x, y, -l/2));
    normals.push_back(glm::vec3(x, y, -1));
  }
  for (int i = 1; i < slices + 1; i++) {
    indices.push_back(0);
    indices.push_back(i % slices + 1);
    indices.push_back(i);
  }
//Top
  position = slices;
  for (int i = 0; i < slices; i++) {
    float theta = 2 * M_PI / slices * i;
    float y = r * std::cos(theta);
    float x = -r * std::sin(theta);
    positions.push_back(glm::vec3(x, y, l));
    normals.push_back(glm::vec3(x, y, 1));
    //    normals.push_back(glm::vec3(x, y, l/2));
  }
  positions.push_back(glm::vec3(0, 0, l));
  normals.push_back(glm::vec3(0, 0, 1));

  for (int i = 1; i < slices + 1; i++) {
    indices.push_back(2 * position);
    indices.push_back(i + position);
    indices.push_back(i % slices + 1 + position);
  }
//  for (int i = 1; i < slices + 1; i++) {
//    indices.push_back(i);
//    indices.push_back(i % slices + 1);
//    indices.push_back(position + i);
//
//    indices.push_back(position + i);
//    indices.push_back(i % slices + 1);
//    indices.push_back(position + i % slices + 1);
//  }
  position += slices + 2;
  // side
  for (int i = 0; i < slices; i++) {
    float theta = 2 * M_PI / slices * i;
    float y = r * std::cos(theta);
    float x = -r * std::sin(theta);
    positions.push_back(glm::vec3(x, y, 0));
    normals.push_back(glm::vec3(x, y, 0));
    positions.push_back(glm::vec3(x, y, l));
    normals.push_back(glm::vec3(x, y, 0));
    //    normals.push_back(glm::vec3(x, y, l/2));
  }

  for (int i = 0; i < slices; i++) {
    indices.push_back(position + 2 * i);
    indices.push_back(position + (2 * (i + 1))%(2*slices));
    indices.push_back(position + 2 * i + 1);

    indices.push_back(position + 2 * i + 1);
    indices.push_back(position + (2 * (i + 1))%(2*slices));
    indices.push_back(position + (2 * (i + 1) + 1)%(2*slices));
  }

}

}
}
}
//
//// Created by joonho on 19.05.17.
////
//
//#include "Cylinder.hpp"
//
//namespace RAI {
//namespace Graphics {
//namespace Obj {
//
//Cylinder::Cylinder(float r,float l) {
//  int slices = 20;
//  int position = 0;
////  bottom
//  positions.push_back(glm::vec3(0, 0, 0));
//  normals.push_back(glm::vec3(0, 0, -1));
//
//  for (int i = 0; i < slices; i++) {
//    float theta = 2 * M_PI / slices * i;
//    float y = r * std::cos(theta);
//    float x = -r * std::sin(theta);
//    positions.push_back(glm::vec3(x, y, 0));
//    normals.push_back(glm::vec3(x, y, -l/2));
//  }
//  for (int i = 1; i < slices + 1; i++) {
//    indices.push_back(0);
//    indices.push_back(i % slices + 1);
//    indices.push_back(i);
//  }
////Top
//  position = slices;
//
//  for (int i = 0; i < slices; i++) {
//    float theta = 2 * M_PI / slices * i;
//    float y = r * std::cos(theta);
//    float x = -r * std::sin(theta);
//    positions.push_back(glm::vec3(x, y, l));
//    normals.push_back(glm::vec3(x, y, l/2));
//  }
//  positions.push_back(glm::vec3(0, 0, l));
//  normals.push_back(glm::vec3(0, 0, 1));
//
//  for (int i = 1; i < slices + 1; i++) {
//    indices.push_back(2* position);
//    indices.push_back(i + position);
//    indices.push_back(i % slices + 1+ position);
//  }
//
//  for (int i = 1; i < slices + 1; i++) {
//    indices.push_back(i);
//    indices.push_back(i % slices + 1);
//    indices.push_back(position+ i);
//
//    indices.push_back(position + i);
//    indices.push_back(i % slices + 1);
//    indices.push_back(position + i % slices + 1);
//  }
//
////
//  position += slices + 1;
//
//
//
//}
//
//}
//}
//}