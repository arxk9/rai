#include "Mesh.hpp"
#include <map>
#include <fstream>
#include <sys/stat.h>
#include <iostream>
#include <imp/vector3d.h>
#include "SDL2/SDL_image.h"

inline bool fileexists (const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}

namespace rai {
namespace Graphics {
namespace Obj {

Mesh::Mesh(const std::string& fileName, float scale) {
  LOG_IF(FATAL, !fileexists(fileName))<< "could not find the mesh file"<<std::endl;
  Assimp::Importer importer;
  const aiScene *scene = importer.ReadFile(fileName.c_str(),
                                           aiProcess_GenSmoothNormals | aiProcess_Triangulate
                                               | aiProcess_CalcTangentSpace | aiProcess_FlipUVs);
  if (scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
    std::cout << "The file wasn't successfuly opened " << fileName << std::endl;
    return;
  }
  scale_ = scale;
  recursiveProcess(scene->mRootNode, scene);
}

Mesh::Mesh(Vertex *vertices, unsigned int numVertices, unsigned int *indicesL, unsigned int numIndices) {
  for (unsigned int i = 0; i < numVertices; i++) {
    positions.push_back(*vertices[i].GetPos());
    texCoords.push_back(*vertices[i].GetTexCoord());
    normals.push_back(*vertices[i].GetNormal());
  }

  for (unsigned int i = 0; i < numIndices; i++)
    indices.push_back(indicesL[i]);
}

Mesh::~Mesh() = default;

void Mesh::recursiveProcess(aiNode *node, const aiScene *scene) {
  //process
  for (int i = 0; i < node->mNumMeshes; i++) {
    aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
    processMesh(mesh, scene);
  }

  //recursion
  for (int i = 0; i < node->mNumChildren; i++) {
    recursiveProcess(node->mChildren[i], scene);
  }
}

void Mesh::processMesh(aiMesh *mesh, const aiScene *scene) {
  aiColor4D col;
  aiMaterial *mat = scene->mMaterials[mesh->mMaterialIndex];
  aiGetMaterialColor(mat, AI_MATKEY_COLOR_DIFFUSE, &col);
  vector3d defaultColor(col.r, col.g, col.b);

  for (int i = 0; i < mesh->mNumVertices; i++) {
    positions.push_back(glm::vec3(mesh->mVertices[i].x*scale_, mesh->mVertices[i].y*scale_, mesh->mVertices[i].z*scale_));
    normals.push_back(glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z));

    //colors
    glm::vec3 color;

    if (mesh->mColors[0]) {
      //!= material color
      color = glm::vec3(mesh->mColors[0][i].r, mesh->mColors[0][i].g, mesh->mColors[0][i].b);
    } else {
      color = glm::vec3(0.5,0.5,0.5);
    }
    colorsCoords.push_back(color);
  }

  for (int i = 0; i < mesh->mNumFaces; i++) {
    aiFace face = mesh->mFaces[i];
    for (int j = 0; j < face.mNumIndices; j++) //0..2
    {
      indices.push_back(face.mIndices[j]);
    }
  }
}

}
}
}