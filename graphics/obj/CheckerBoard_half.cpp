//
// Created by jhwangbo on 17. 4. 29.
//

//
// Created by jhwangbo on 17. 4. 29.
//

#include "CheckerBoard_half.hpp"

namespace RAI {
namespace Graphics {
namespace Obj {

CheckerBoard_half::CheckerBoard_half(int gridSize,
                           float width,
                           float length,
                           std::vector<float> color) {

  for(int i=-width/2/gridSize; i<width/2/gridSize; i++)
    for(int j=-length/2/gridSize; j<length/2/gridSize; j++){
      positions.push_back(glm::vec3(j*gridSize, i*gridSize, 0));
      normals.push_back(glm::vec3(0,0,1));
    }

  color_ = color;

  int widthPoints = width/2/gridSize+width/2/gridSize;
  int heightPoints = length/2/gridSize+length/2/gridSize;

  for(int i=0; i<widthPoints-1; i++)
    for(int j=0; j<heightPoints-1; j+=2){
      if(i % 2 == 0 && j == 0){
        j -= 1;
        continue;
      }

      indices.push_back(i*heightPoints+j);
      indices.push_back(i*heightPoints+j+1);
      indices.push_back((i+1)*heightPoints+j);

      indices.push_back(i*heightPoints+j+1);
      indices.push_back((i+1)*heightPoints+j+1);
      indices.push_back((i+1)*heightPoints+j);
    }
}

}
}
}

