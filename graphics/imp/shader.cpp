//
// Created by jhwangbo on 17. 4. 30.
//

#include <glog/logging.h>
#include "shader.hpp"

namespace rai {
namespace Graphics {

std::string Shader::LoadShader(const std::string &fileName) {
  std::ifstream file;
  file.open((fileName).c_str());

  std::string output;
  std::string line;

  if (file.is_open()) {
    while (file.good()) {
      getline(file, line);
      output.append(line + "\n");
    }
  } else {
    LOG(FATAL)<< "Unable to load shader: " << fileName << std::endl;
  }

  return output;
}

void Shader::CheckShaderError(GLuint shader, GLuint flag, bool isProgram, const std::string &errorMessage) {
  GLint success = 0;
  GLchar error[1024] = {0};

  if (isProgram)
    glGetProgramiv(shader, flag, &success);
  else
    glGetShaderiv(shader, flag, &success);

  if (success == GL_FALSE) {
    if (isProgram)
      glGetProgramInfoLog(shader, sizeof(error), NULL, error);
    else
      glGetShaderInfoLog(shader, sizeof(error), NULL, error);

    LOG(FATAL) << errorMessage << ": '" << error << "'";
  }
}

GLuint Shader::CreateShader(const std::string &text, unsigned int type) {
  GLuint shader = glCreateShader(type);

  LOG_IF(FATAL, shader == 0) << "Error compiling shader type " << type << std::endl;

  const GLchar *p[1];
  p[0] = text.c_str();
  GLint lengths[1];
  lengths[0] = text.length();

  glShaderSource(shader, 1, p, lengths);
  glCompileShader(shader);

  CheckShaderError(shader, GL_COMPILE_STATUS, false, "Error compiling shader!");

  return shader;
}

}
}