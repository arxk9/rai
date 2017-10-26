#include "display.h"
#include <GL/glew.h>
#include <iostream>
#include <glog/logging.h>

namespace RAI {
namespace Graphics {
Display::Display(int width, int height, const std::string &title) {
  SDL_Init(SDL_INIT_EVERYTHING);

  SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
  SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
  SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
  SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);
  SDL_GL_SetAttribute(SDL_GL_BUFFER_SIZE, 32);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
  SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
  SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 2);

  SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1");

  m_window =
      SDL_CreateWindow(title.c_str(), 0, 0, width, height, SDL_WINDOW_OPENGL);
  LOG_IF(FATAL, m_window ==NULL) << SDL_GetError();

  m_glContext = SDL_GL_CreateContext(m_window);
  LOG_IF(FATAL, m_glContext ==NULL) << SDL_GetError();

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glewExperimental = GL_TRUE;
  GLenum res = glewInit();
  LOG_IF(FATAL, res != GLEW_OK) << "Glew failed to initialize!" << " Error: "<<glewGetErrorString(res) << std::endl;

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);

}

Display::~Display() {
  SDL_GL_DeleteContext(m_glContext);
  SDL_DestroyWindow(m_window);
  SDL_Quit();
}

void Display::Clear(float r, float g, float b, float a) {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glClearColor(r, g, b, a);
}

void Display::SwapBuffers() {
  SDL_GL_SwapWindow(m_window);
}
}
}