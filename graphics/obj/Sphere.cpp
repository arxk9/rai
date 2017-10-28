//
// Created by jhwangbo on 17. 4. 29.
//

#include "Sphere.hpp"

namespace rai {
namespace Graphics {
namespace Obj {

Sphere::Sphere(float radius, int rings) {
  float const R = 1. / (float) (rings - 1);
  float const S = 1. / (float) (rings - 1);
  int r, s;

  positions.resize(rings * rings * 3);
  normals.resize((rings-1) * (rings-1) * 3);
  std::vector<glm::vec3>::iterator v = positions.begin();
  std::vector<glm::vec3>::iterator n = normals.begin();

  for (r = 0; r < rings; r++)
    for (s = 0; s < rings; s++) {
      float const y = sin(-M_PI_2 + M_PI * r * R);
      float const x = cos(2 * M_PI * s * S) * sin(M_PI * r * R);
      float const z = sin(2 * M_PI * s * S) * sin(M_PI * r * R);

      v->x = x * radius;
      v->y = y * radius;
      v->z = z * radius;
      v++;

      n->x = x * radius;
      n->y = y * radius;
      n->z = z * radius;
      n++;
    }

  indices.resize((rings-1) * (rings-1) * 6);
  std::vector<unsigned>::iterator i = indices.begin();
  unsigned idx=0;
  for (r = 0; r < rings - 1; r++)
    for (s = 0; s < rings - 1; s++) {
      indices[idx] = r * rings + s; idx++;
      indices[idx] = (r + 1) * rings + (s + 1); idx++;
      indices[idx] = r * rings + (s + 1); idx++;


      indices[idx] = r * rings + s; idx++;
      indices[idx] = (r + 1) * rings + s; idx++;
      indices[idx] = (r + 1) * rings + (s + 1); idx++;

    }
}

}
}
}

