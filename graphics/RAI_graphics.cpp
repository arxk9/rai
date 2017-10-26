//
// Created by jhwangbo on 17. 4. 28.
//

#include "RAI_graphics.hpp"
#include "glog/logging.h"
#include <FreeImage.h>
#include <thread>

namespace rai {
namespace Graphics {

typedef void *(RAI_graphics::*Thread2Ptr)(void *);
typedef void *(*PthreadPtr)(void *);

RAI_graphics::RAI_graphics(int windowWidth, int windowHeight) {
  windowWidth_ = windowWidth;
  windowHeight_ = windowHeight;
}

RAI_graphics::~RAI_graphics() {

}

void RAI_graphics::start() {
  Thread2Ptr t = &RAI_graphics::loop;
  PthreadPtr p = *(PthreadPtr *) &t;
  pthread_create(&mainloopThread, 0, p, this);
  mtxLoop.lock();
}

void RAI_graphics::end() {
  mtxLoop.unlock();
  pthread_join(mainloopThread, NULL);
}

void* RAI_graphics::loop(void *obj){
  display = new Display(windowWidth_, windowHeight_, "RAI simulator");
  camera = new Camera(glm::vec3(0.0f, 0.0f, 5.0f), 70.0f, (float) windowWidth_ / (float) windowHeight_, 0.1f, 1000.0f);
  shader_basic = new Shader_basic();
  shader_background = new Shader_background();
  light = new Light();

  while(true){
    if( mtx.try_lock() ) {
      if(mtxLoop.try_lock()) break;
      watch.start();
      mtxinit.lock();
      init();
      mtxinit.unlock();
      draw();
      mtx.unlock();
      double elapse = watch.measure();
      if(terminate) break;
      usleep(std::max((1.0 / FPS_ - elapse) * 1e6, 0.0));
    }
  }

  for (auto* sob: supObjs_)
    sob->destroy();

  for (auto* ob: objs_)
    ob->destroy();

  delete display;
  delete camera;
  delete shader_background;
  delete shader_basic;
}

void RAI_graphics::init() {
  if (background && backgroundChanged) background->init();
  backgroundChanged = false;

  mtxLight.lock();
  if (lightPropChanged) {
    light->setDiffuse(lightProp.diff_light);
    light->setAmbient(lightProp.amb_light);
    light->setSpecular(lightProp.spec_light);
    light->setPosition(lightProp.pos_light);
    lightPropChanged = false;
  }
  mtxLight.unlock();

  mtxCamera.lock();
  if (cameraPropChanged) {
    if (cameraProp.toFollow)
      camera->follow(cameraProp.toFollow, cameraProp.relativeDist);
    cameraPropChanged = false;
  }
  mtxCamera.unlock();

  for (auto* sob: added_supObjs_) {
    sob->init();
    supObjs_.push_back(sob);
  }

  for (auto *ob: added_objs_) {
    ob->init();
    objs_.push_back(ob);
  }

  for (auto sh: added_shaders_)
    switch (sh) {
      case RAI_SHADER_BASIC: shaders_.push_back(shader_basic);
        break;
      default: shaders_.push_back(shader_basic);
        break;
    }

  for (auto *ob: tobeRemoved_objs_) {
    ptrdiff_t pos = find(objs_.begin(), objs_.end(), ob) - objs_.begin();
    objs_.erase(objs_.begin() + pos);
    shaders_.erase(shaders_.begin() + pos);
  }

  for (auto *sob: tobeRemoved_supObjs_) {
    ptrdiff_t pos = find(supObjs_.begin(), supObjs_.end(), sob) - supObjs_.begin();
    supObjs_.erase(supObjs_.begin() + pos);
  }

  added_objs_.clear();
  added_shaders_.clear();
  added_supObjs_.clear();
  tobeRemoved_supObjs_.clear();
  tobeRemoved_objs_.clear();
}

void RAI_graphics::draw() {
  SDL_PollEvent(&e);
  camera->Control(e);
  display->Clear(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
  camera->update();

  if (background) {
    shader_background->Bind();
    shader_background->Update(camera, light, background);
    background->draw();
    shader_background->UnBind();
  }

  for (auto *sob: supObjs_)
    if(sob->isVisible()) sob->draw(camera, light);

  for (int i = 0; i < objs_.size(); i++) {
    if(!objs_[i]->isVisible()) continue;
    shaders_[i]->Bind();
    shaders_[i]->Update(camera, light, objs_[i]);
    objs_[i]->draw();
    shaders_[i]->UnBind();
  }

  if (saveSnapShot) {
    if( imageCounter < 2e3) {
      areThereimagesTosave = true;
      std::string imageFileName = std::to_string(imageCounter++);
      while (imageFileName.length() < 7)
        imageFileName = "0" + imageFileName;
      GLubyte *pixels = new GLubyte[3 * windowWidth_ * windowHeight_];
      glReadPixels(0, 0, windowWidth_, windowHeight_, GL_BGR, GL_UNSIGNED_BYTE, pixels);
      FIBITMAP *image = FreeImage_ConvertFromRawBits(pixels, windowWidth_, windowHeight_, 3 * windowWidth_, 24,
                                                     0xFF0000, 0x00FF00, 0x0000FF, false);
      FreeImage_Save(FIF_PNG, image, (image_dir + "/" + imageFileName + ".png").c_str(), 0);
      FreeImage_Unload(image);
      delete[] pixels;
    } else {
      LOG(FATAL) << "You are saving too many frames. RAI shutting down to prevent possible system breakdown";
    }
  }

  display->SwapBuffers();
}

void RAI_graphics::addObject(Obj::Object *obj, ShaderType type) {
  std::lock_guard<std::mutex> guard(mtxinit);
  LOG_IF(FATAL, !obj) << "the object is not created yet";
  added_objs_.push_back(obj);
  added_shaders_.push_back(type);
}

void RAI_graphics::addSuperObject(Obj::SuperObject *obj) {
  std::lock_guard<std::mutex> guard(mtxinit);
  LOG_IF(FATAL, !obj) << "the object is not created yet";
  added_supObjs_.push_back(obj);
}

void RAI_graphics::addBackground(Obj::Background *back) {
  std::lock_guard<std::mutex> guard(mtxinit);
  backgroundChanged = true;
  LOG_IF(FATAL, !back) << "the object is not created yet";
  background = back;
}

void RAI_graphics::removeObject(Obj::Object *obj) {
  std::lock_guard<std::mutex> guard(mtxinit);
  tobeRemoved_objs_.push_back(obj);
}

void RAI_graphics::removeSuperObject(Obj::SuperObject *obj) {
  std::lock_guard<std::mutex> guard(mtxinit);
  tobeRemoved_supObjs_.push_back(obj);
}

void RAI_graphics::setBackgroundColor(float r, float g, float b, float a) {
  clearColor[0] = r;
  clearColor[1] = g;
  clearColor[2] = b;
  clearColor[3] = a;
}

void RAI_graphics::setLightProp(LightProp& prop) {
  std::lock_guard<std::mutex> guard(mtxLight);
  lightProp = prop;
  lightPropChanged = true;
}

void RAI_graphics::setCameraProp(CameraProp& prop) {
  std::lock_guard<std::mutex> guard(mtxCamera);
  cameraProp = prop;
  cameraPropChanged = true;
}

void RAI_graphics::savingSnapshots(std::string logDirectory, std::string fileName) {
  mtx.lock();
  image_dir = logDirectory;
  imageCounter = 0;
  videoFileName = fileName;
  saveSnapShot = true;
  mtx.unlock();
}

void RAI_graphics::images2Video() {
  std::cout<<"saving video. This might take a few seconds"<<std::endl;
  std::lock_guard<std::mutex> guard(mtxCamera);
  saveSnapShot = false;
  if(!areThereimagesTosave) return;
  areThereimagesTosave = false;
  Thread2Ptr t = &RAI_graphics::images2Video_inThread;
  PthreadPtr p = *(PthreadPtr *) &t;
  pthread_t tid;
  if (pthread_create(&tid, 0, p, this) == 0)
    pthread_detach(tid);

  while(mtx.try_lock())
    mtx.unlock();
}

void *RAI_graphics::images2Video_inThread(void *obj) {
  std::lock_guard<std::mutex> guard(mtx);
  std::string command = "ffmpeg -r 60 -i " + image_dir + "/%07d.png -s 800x600 -c:v libx264 -crf 5 " + image_dir + "/" + videoFileName + ".mp4 >nul 2>&1";
  system(command.c_str());
  command = "rm -rf " + image_dir + "/*.png";
  system(command.c_str());
  return NULL;
}

const Uint8* RAI_graphics::keyboard() {
  return SDL_GetKeyboardState(NULL);
}

const RAI_graphics::MouseInput* RAI_graphics::mouse(){
  SDL_Event e;
  SDL_PollEvent( &e );
  mouseInput.wheel = e.wheel;
  mouseInput.leftB = e.button.button;
  Uint32 mbuttonState = SDL_GetMouseState(&mouseInput.x, &mouseInput.y);

  if (mbuttonState==SDL_BUTTON_LEFT) mouseInput.leftB = true;
  else mouseInput.leftB = false;

  if (mbuttonState==SDL_BUTTON_RIGHT) mouseInput.rightB = true;
  else mouseInput.rightB = false;

  if (mbuttonState==SDL_BUTTON_MIDDLE) mouseInput.middleB = true;
  else mouseInput.middleB = false;

  return &mouseInput;
}

}
}