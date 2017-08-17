//
// Created by jhwangbo on 17. 4. 30.
//

#ifndef RAI_BACKGROUND_HPP
#define RAI_BACKGROUND_HPP

#include "Object.hpp"
#include <vector>
#include "math.h"
#include "TypeDef.hpp"

namespace RAI {
namespace Graphics {
namespace Obj {

class Background {

 public:

  Background(std::string filename = "", std::string filetype = "jpg");
  void init();
  void draw();

 private:
  std::vector<std::string> faces;
  GLuint loadCubemap(std::vector<std::string>& faces);
  GLuint skyboxVAO, skyboxVBO;
  GLuint cubemapTexture;
  GLfloat skyboxVertices[108] = {
      // Positions
      -500.0f,  500.0f, -500.0f,
      -500.0f, -500.0f, -500.0f,
      500.0f, -500.0f, -500.0f,
      500.0f, -500.0f, -500.0f,
      500.0f,  500.0f, -500.0f,
      -500.0f,  500.0f, -500.0f,

      -500.0f, -500.0f,  500.0f,
      -500.0f, -500.0f, -500.0f,
      -500.0f,  500.0f, -500.0f,
      -500.0f,  500.0f, -500.0f,
      -500.0f,  500.0f,  500.0f,
      -500.0f, -500.0f,  500.0f,

      500.0f, -500.0f, -500.0f,
      500.0f, -500.0f,  500.0f,
      500.0f,  500.0f,  500.0f,
      500.0f,  500.0f,  500.0f,
      500.0f,  500.0f, -500.0f,
      500.0f, -500.0f, -500.0f,

      -500.0f, -500.0f,  500.0f,
      -500.0f,  500.0f,  500.0f,
      500.0f,  500.0f,  500.0f,
      500.0f,  500.0f,  500.0f,
      500.0f, -500.0f,  500.0f,
      -500.0f, -500.0f,  500.0f,

      -500.0f,  500.0f, -500.0f,
      500.0f,  500.0f, -500.0f,
      500.0f,  500.0f,  500.0f,
      500.0f,  500.0f,  500.0f,
      -500.0f,  500.0f,  500.0f,
      -500.0f,  500.0f, -500.0f,

      -500.0f, -500.0f, -500.0f,
      -500.0f, -500.0f,  500.0f,
      500.0f, -500.0f, -500.0f,
      500.0f, -500.0f, -500.0f,
      -500.0f, -500.0f,  500.0f,
      500.0f, -500.0f,  500.0f
  };

};
}
}
}

#endif //RAI_BACKGROUND_HPP
