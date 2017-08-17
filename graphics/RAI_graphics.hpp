//
// Created by jhwangbo on 17. 4. 28.
//

#ifndef PROJECT_RAI_GRAPHICS_HPP
#define PROJECT_RAI_GRAPHICS_HPP
#include <obj/Background.hpp>
#include "obj/Object.hpp"
#include "obj/SuperObject.hpp"
#include "imp/display.h"
#include "imp/shader_basic.h"
#include "imp/shader_background.hpp"
#include "StopWatch.hpp"
#include <mutex>
#include "RAI_keyboard.hpp"

namespace RAI {
namespace Graphics {

class RAI_graphics {
 public:

  enum ShaderType {
    RAI_SHADER_BASIC
  };

  struct LightProp {
    std::vector<float>  pos_light = {-100.0,0.0,10.0},
                        amb_light = {0.5, 0.5, 0.5},
                        diff_light = {1, 1, 1},
                        spec_light = {0.7, 0.7, 0.7};
  };

  struct MouseInput {
    int x, y;
    bool leftB, rightB, middleB;
    SDL_MouseWheelEvent wheel;
  };

  struct CameraProp {
    Obj::Object *toFollow = nullptr;
    Eigen::Vector3d relativeDist = Eigen::Vector3d::Constant(1);
  };

  RAI_graphics(int windowWidth, int windowHeight);
  ~RAI_graphics();
  void start();
  void end();

  void addObject(Obj::Object *obj, ShaderType type = RAI_SHADER_BASIC);
  void addSuperObject(Obj::SuperObject *obj);
  void addBackground(Obj::Background *back);
  void setFPS(double FPS) { FPS_ = FPS; }

  void removeObject(Obj::Object *obj);
  void removeSuperObject(Obj::SuperObject *obj);
  void setBackgroundColor(float r, float g, float b, float a);
  void setLightProp(LightProp &prop);
  void setCameraProp(CameraProp &prop);

  void savingSnapshots(std::string logDirectory, std::string fileName);
  void images2Video();

  const Uint8* keyboard();
  const MouseInput* mouse();

 private:
  void *loop(void *obj);
  void init();
  void draw();
  void *images2Video_inThread(void *obj);

  Obj::Background *background = nullptr;
  bool backgroundChanged;

  std::vector<Obj::Object *> objs_;
  std::vector<Obj::SuperObject *> supObjs_;

  std::vector<Obj::Object *> added_objs_;
  std::vector<Obj::SuperObject *> added_supObjs_;

  std::vector<Obj::Object *> tobeRemoved_objs_;
  std::vector<Obj::SuperObject *> tobeRemoved_supObjs_;

  Display *display = nullptr;
  Shader_basic *shader_basic = nullptr;
  Shader_background *shader_background = nullptr;
  std::vector<Shader *> shaders_;
  std::vector<ShaderType> added_shaders_;

  unsigned imageCounter = 0;
  bool areThereimagesTosave = false;
  bool saveSnapShot = false;
  Camera *camera = nullptr;
  Light *light = nullptr;
  int windowWidth_, windowHeight_;
  SDL_Event e;
  bool freeCamMode;
  float clearColor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  double FPS_ = 60.0;
  std::string image_dir, videoFileName;
  StopWatch watch;
  bool terminate = false;
  std::mutex mtx;           // mutex for critical section
  std::mutex mtxLoop;           // mutex for critical section
  std::mutex mtxinit;           // mutex for critical section
  std::mutex mtxLight;
  std::mutex mtxCamera;

  pthread_t mainloopThread;
  MouseInput mouseInput;
  LightProp lightProp;
  CameraProp cameraProp;
  bool cameraPropChanged;
  bool lightPropChanged;

};

}
}

#endif //PROJECT_RAI_GRAPHICS_HPP
