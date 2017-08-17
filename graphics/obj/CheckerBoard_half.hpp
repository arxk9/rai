//
// Created by jhwangbo on 02.12.16.
//

#ifndef RAI_CHECKERBOARD_HALF_HPP
#define RAI_CHECKERBOARD_HALF_HPP

#include "Object.hpp"
#include <vector>
#include "math.h"
#include "TypeDef.hpp"

namespace RAI {
namespace Graphics {
namespace Obj {

class CheckerBoard_half : public Object {

 public:

  CheckerBoard_half(int gridSize, float width, float length, std::vector<float> color1);

};

}
}
}

#endif //RAI_CHECKERBOARD_HPP
