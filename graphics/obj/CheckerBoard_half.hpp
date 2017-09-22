//
// Created by jhwangbo on 02.12.16.
//

#ifndef RAI_CHECKERBOARD_HALF_HPP
#define RAI_CHECKERBOARD_HALF_HPP

#include "Object.hpp"
#include <vector>
#include "math.h"
#include "rai/common/TypeDef.hpp"
#include "rai/RAI_Vector.hpp"

namespace rai {
namespace Graphics {
namespace Obj {

class CheckerBoard_half : public Object {

 public:

  CheckerBoard_half(int gridSize, float width, float length, rai::Vector<float> color1);

};

}
}
}

#endif //RAI_CHECKERBOARD_HPP
