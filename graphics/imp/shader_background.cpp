//
// Created by jhwangbo on 17. 4. 30.
//
#include <obj/Background.hpp>
#include <glog/logging.h>
#include "shader_background.hpp"

namespace rai {
namespace Graphics {

Shader_background::Shader_background() {
  m_program = glCreateProgram();
  m_shaders[0] = CreateShader(LoadShader(std::string(getenv("RAI_ROOT")) + "/graphics/res/cubeMapShader.vs"), GL_VERTEX_SHADER);
  m_shaders[1] = CreateShader(LoadShader(std::string(getenv("RAI_ROOT")) + "/graphics/res/cubeMapShader.fs"), GL_FRAGMENT_SHADER);

  for (unsigned int i = 0; i < NUM_SHADERS; i++)
    glAttachShader(m_program, m_shaders[i]);

  glLinkProgram(m_program);
  CheckShaderError(m_program, GL_LINK_STATUS, true, "Error linking shader program");

  glValidateProgram(m_program);
  CheckShaderError(m_program, GL_LINK_STATUS, true, "Invalid shader program");
}

Shader_background::~Shader_background() {
  for (unsigned int i = 0; i < NUM_SHADERS; i++) {
    glDetachShader(m_program, m_shaders[i]);
    glDeleteShader(m_shaders[i]);
  }
  glDeleteProgram(m_program);
}

void Shader_background::Bind() {
  glUseProgram(m_program);
}

void Shader_background::UnBind() {
  glUseProgramObjectARB(0);
}

void Shader_background::Update(Camera *camera, Light *light, Obj::Object* obj){
  LOG(FATAL) << "Shader_background only works with a background object"<<std::endl;
}

void Shader_background::Update(Camera *camera, Light *light, Obj::Background* obj) {
  glm::mat4 MVP;
  camera->GetVP(MVP);
  glm::mat4 Normal;
  glUniformMatrix4fv(glGetUniformLocation(m_program, "MVP"), 1, GL_FALSE, &MVP[0][0]);
  glUniformMatrix4fv(glGetUniformLocation(m_program, "Normal"), 1, GL_FALSE, &Normal[0][0]);
  glUniform1i(glGetUniformLocation(m_program, "skybox"), 0);
}

}
}