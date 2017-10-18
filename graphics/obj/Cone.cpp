//
// Created by joonho on 19.05.17.
//

#include "Cone.hpp"

namespace rai {
namespace Graphics {
namespace Obj {

Cone::Cone(float r,float l) {
  int slices = 20;
//  bottom
  positions.push_back(glm::vec3(0, 0, 0));
  normals.push_back(glm::vec3(0, 0, -1));

  for (int i = 0; i < slices; i++) {
    float theta = 2 * M_PI / slices * i;
    float y = r * std::cos(theta);
    float x = -r * std::sin(theta);
    positions.push_back(glm::vec3(x, y, 0));
    normals.push_back(glm::vec3(x, y, -l/(sqrt(r*r + l*l)+r)));
  }
  for (int i = 1; i < slices + 1; i++) {
    indices.push_back(0);
    indices.push_back(i % slices + 1);
    indices.push_back(i);
  }
//Top
  positions.push_back(glm::vec3(0, 0, l));
  normals.push_back(glm::vec3(0, 0, 1));

  for (int i = 1; i < slices + 1; i++) {
    indices.push_back(slices + 1);
    indices.push_back(i);
    indices.push_back(i % slices + 1);
  }
}

}
}
}
