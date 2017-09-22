#ifndef OBJ_LOADER_H_INCLUDED
#define OBJ_LOADER_H_INCLUDED

#include <glm/glm.hpp>
#include <vector>
#include <string>
#include "glog/logging.h"
#include "rai/RAI_Vector.hpp"

namespace rai {
namespace Graphics {
struct OBJIndex {
  unsigned int vertexIndex;
  unsigned int uvIndex;
  unsigned int normalIndex;

  bool operator<(const OBJIndex &r) const { return vertexIndex < r.vertexIndex; }
};

class IndexedModel {
 public:
  rai::Vector<glm::vec3> positions;
  rai::Vector<glm::vec2> texCoords;
  rai::Vector<glm::vec3> normals;
  rai::Vector<unsigned int> indices;

  void CalcNormals();
};

class OBJModel {
 public:
  rai::Vector<OBJIndex> OBJIndices;
  rai::Vector<glm::vec3> vertices;
  rai::Vector<glm::vec2> uvs;
  rai::Vector<glm::vec3> normals;
  bool hasUVs;
  bool hasNormals;

  OBJModel(const std::string &fileName);

  IndexedModel ToIndexedModel();
 private:
  unsigned int FindLastVertexIndex(const rai::Vector<OBJIndex *> &indexLookup,
                                   const OBJIndex *currentIndex,
                                   const IndexedModel &result);
  void CreateOBJFace(const std::string &line);

  glm::vec2 ParseOBJVec2(const std::string &line);
  glm::vec3 ParseOBJVec3(const std::string &line);
  OBJIndex ParseOBJIndex(const std::string &token, bool *hasUVs, bool *hasNormals);
};
}
}
#endif // OBJ_LOADER_H_INCLUDED
