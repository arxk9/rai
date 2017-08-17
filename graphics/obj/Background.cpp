//
// Created by jhwangbo on 17. 4. 30.
//

#include <iostream>
#include <sys/stat.h>
#include "Background.hpp"
#include "SOIL/SOIL.h"

namespace RAI {
namespace Graphics {
namespace Obj {

Background::Background(std::string filename, std::string filetype) {
  std::cout<<std::string(getenv("RAI_ROOT"))+ "/graphics/res/"<<filename<<std::endl;
  faces.push_back(std::string(getenv("RAI_ROOT"))+ "/graphics/res/" +filename + "_right."+filetype);
  faces.push_back(std::string(getenv("RAI_ROOT"))+ "/graphics/res/" +filename + "_left."+filetype);
  faces.push_back(std::string(getenv("RAI_ROOT"))+ "/graphics/res/" +filename + "_front."+filetype);
  faces.push_back(std::string(getenv("RAI_ROOT"))+ "/graphics/res/" +filename + "_back."+filetype);
  faces.push_back(std::string(getenv("RAI_ROOT"))+ "/graphics/res/" +filename + "_top."+filetype);
  faces.push_back(std::string(getenv("RAI_ROOT"))+ "/graphics/res/" +filename + "_bottom."+filetype);
}

void Background::init() {
  cubemapTexture = loadCubemap(faces);
  // Setup skybox VAO
  glGenVertexArrays(1, &skyboxVAO);
  glGenBuffers(1, &skyboxVBO);
  glBindVertexArray(skyboxVAO);
  glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices[0])*108, skyboxVertices, GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
  glBindVertexArray(0);
}

GLuint Background::loadCubemap(std::vector<std::string>& faces)
{
  GLuint textureID;
  glGenTextures(1, &textureID);
  glActiveTexture(GL_TEXTURE0);

  int width,height;
  unsigned char* image;

  glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);
  for(GLuint i = 0; i < faces.size(); i++)
  {
    image = SOIL_load_image(faces[i].c_str(), &width, &height, 0, SOIL_LOAD_RGB);
    glTexImage2D( GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0,
        GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
  }
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

  return textureID;
}

void Background::draw() {
  glDepthMask(GL_FALSE);
  glBindVertexArray(skyboxVAO);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
  glDrawArrays(GL_TRIANGLES, 0, 36);
  glBindVertexArray(0);
  glDepthMask(GL_TRUE);
}

}
}
}