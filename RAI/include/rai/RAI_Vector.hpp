//
// Created by joonho on 9/19/17.
//

#ifndef RAI_RAI_VECTOR_HPP
#define RAI_RAI_VECTOR_HPP

#include <string>
#include <vector>
#include <Eigen/StdVector>
#include "tensorflow/core/framework/tensor.h"


namespace rai {
template<typename T>
class Vector: public std::vector<T,Eigen::aligned_allocator<T>> {
typedef std::vector<T,Eigen::aligned_allocator<T>> stdVector;
  using stdVector::stdVector;
};

// Full specialization
template<>
class Vector<std::string>: public std::vector<std::string> {
  typedef std::vector<std::string> stdVector;
  using stdVector::stdVector;
};

template<>
class Vector<tensorflow::Tensor>: public std::vector<tensorflow::Tensor> {
  typedef std::vector<tensorflow::Tensor> stdVector;
  using stdVector::stdVector;
};

template<>
class Vector<std::pair<std::string, tensorflow::Tensor>> : public std::vector<std::pair<std::string, tensorflow::Tensor>> {
  typedef std::vector<std::pair<std::string, tensorflow::Tensor>> stdVector;
  using stdVector::stdVector;
};


}
#endif //RAI_RAI_VECTOR_HPP
