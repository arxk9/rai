//
// Created by jhwangbo on 01.05.17.
//

#ifndef RAI_CHECKERBOARD_HPP
#define RAI_CHECKERBOARD_HPP

#include "SuperObject.hpp"
#include "CheckerBoard_half.hpp"
#include "rai/RAI_Vector.hpp"

namespace rai {
namespace Graphics {
namespace Obj {

class CheckerBoard : public SuperObject {

 public:

  CheckerBoard(int gridSize, float width, float length, rai::Vector<float> color1={0.1,0.1,0.1}, rai::Vector<float> color2={0.9,0.9,0.9});
  ~CheckerBoard();
  void init();
  void destroy();

  CheckerBoard_half board1, board2;
};

}
}
}


#endif //RAI_CHECKERBOARD_HPP
