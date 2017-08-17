//
// Created by jhwangbo on 17. 4. 30.
//

#ifndef RAI_SHADER_HPP
#define RAI_SHADER_HPP

#include <string>
#include <GL/glew.h>
#include "transform.h"
#include "Light.hpp"
#include "../obj/Object.hpp"
#include "camera.h"
#include <iostream>
#include <fstream>

namespace RAI {
namespace Graphics {
class Shader {

 public:
  virtual void Bind() = 0;
  virtual void UnBind() = 0;
  virtual void Update(Camera *camera, Light *light, Obj::Object* obj) = 0;

 protected:
  std::string LoadShader(const std::string &fileName);
  void CheckShaderError(GLuint shader, GLuint flag, bool isProgram, const std::string &errorMessage);
  GLuint CreateShader(const std::string &text, unsigned int type);
  GLuint m_program;

};
}
}
#endif //RAI_SHADER_HPP
