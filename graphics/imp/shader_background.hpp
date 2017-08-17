//
// Created by jhwangbo on 17. 4. 30.
//

#ifndef RAI_SHADER_BACKGROUND_HPP
#define RAI_SHADER_BACKGROUND_HPP
#include "shader.hpp"

namespace RAI {
namespace Graphics {
class Shader_background : public Shader {

 public:
  Shader_background();
  ~Shader_background();

  void Bind();
  void UnBind();
  void Update(Camera *camera, Light *light, Obj::Background* obj);
  void Update(Camera *camera, Light *light, Obj::Object* obj);

 private:
  static const unsigned int NUM_SHADERS = 2;
  static const unsigned int NUM_UNIFORMS = 1;

  GLuint m_shaders[NUM_SHADERS];
  GLuint m_uniforms[NUM_UNIFORMS];
};
}
}
#endif //RAI_SHADER_BACKGROUND_HPP
