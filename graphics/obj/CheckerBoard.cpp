//
// Created by jhwangbo on 01.05.17.
//

#include "CheckerBoard.hpp"

namespace rai {
namespace Graphics {
namespace Obj {

CheckerBoard::CheckerBoard(int gridSize, float width, float length, std::vector<float> color1, std::vector<float> color2):
board1(gridSize, width, length, color1), board2(gridSize, width, length, color2){
  objs.push_back(&board1);
  objs.push_back(&board2);
  Eigen::Vector3d pos; pos<<gridSize, 0,0;
  board2.setPos(pos);
}

CheckerBoard::~CheckerBoard(){
}

void CheckerBoard::init(){
  for(auto* ob: objs)
    ob->init();
  shader = new Shader_basic;
}

void CheckerBoard::destroy(){
  for(auto* ob: objs)
    ob->destroy();
  delete shader;
}

}
}
}