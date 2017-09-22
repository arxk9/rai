//
// Created by jhwangbo on 12/08/17.
//

#ifndef RAI_RAI_TENSOR_HPP
#define RAI_RAI_TENSOR_HPP

#include "vector"
#include "string"
#include "tensorflow/core/public/session.h"
#include "Eigen/Core"
#include <algorithm>
#include "glog/logging.h"
#include <boost/utility/enable_if.hpp>
#include <Eigen/StdVector>
#include "RAI_Vector.hpp"
/* RAI tensor follows EIGEN Tensor indexing
 * The data is stored in tensorflow tensor
 * It provides an interface to Eigen::Tensor and Eigen::Matrix/Vector
 *
 *
 *
 *
 *
 */

namespace rai {

template<typename Dtype, int NDim>
class Tensor {

  typedef Eigen::Map<Eigen::Matrix<Dtype, -1, -1> > EigenMat;
  typedef Eigen::TensorMap<Eigen::Tensor<Dtype, NDim>, Eigen::Aligned> EigenTensor;

 public:

  // empty constructor. Resize has to be called before use
  Tensor() { setDataType(); }

  // empty data constructor
  Tensor(const rai::Vector<int> dim, const std::string name = "") {
    init(dim, name);
  }

///Eigen Tensor constructor is abigous with rai::Vector<int> constructor ...
//  // copy constructor from Eigen Tensor
//  Tensor(const Eigen::Tensor<Dtype, NDim> &etensor, const std::string name = "") {
//    auto dims = etensor.dimensions();
//    rai::Vector<int> dim(dims.size());
//    for (int i = 0; i < dims.size(); i++)
//      dim[i] = dims[i];
//    Tensor(dim, name);
//    std::memcpy(data_->flat<Dtype>().data(), etensor.data(), sizeof(Dtype) * etensor.size());
//  }

  // copy construct from Eigen Matrix
  template<int Rows, int Cols>
  Tensor(const Eigen::Matrix<Dtype, Rows, Cols> &emat, const std::string name = "") {
    LOG_IF(FATAL, NDim != 2) << "Specify the reshape";
    rai::Vector<int> dim = {emat.rows(), emat.cols()};
    init(dim, name);
    std::memcpy(namedTensor_.second.flat<Dtype>().data(), emat.data(), sizeof(Dtype) * emat.size());
  }

  // this constructor is used when the resulting tensor dim is not 2D
  template<int Rows, int Cols>
  Tensor(const Eigen::Matrix<Dtype, Rows, Cols> &emat, rai::Vector<int> dim, const std::string name = "") {
    init(dim, name);
    LOG_IF(FATAL, emat.size() != size_) << "size mismatch";
    std::memcpy(namedTensor_.second.flat<Dtype>().data(), emat.data(), sizeof(Dtype) * emat.size());
  }

  ////////////////////////////
  /////// casting methods ////
  ////////////////////////////
  operator std::pair<std::string, tensorflow::Tensor>() {
    return namedTensor_;
  };

  operator tensorflow::Tensor() {
    return namedTensor_.second;
  };

  template<int Rows, int Cols>
  operator Eigen::Matrix<Dtype, Rows, Cols>() {
    LOG_IF(FATAL, Rows != dim_[0] || Cols != dim_[1]) << "dimension mismatch";
    EigenMat mat(namedTensor_.second.flat<Dtype>().data(), dim_[0], dim_[1]);
    return mat;
  };

  operator EigenTensor() {
    EigenTensor mat(namedTensor_.second.flat<Dtype>().data(), esizes_);
    return mat;
  }

  ////////////////////////////
  /// Eigen Methods mirror ///
  ////////////////////////////
  typename EigenMat::ColXpr col(int colId) {
    LOG_IF(FATAL, dim_.size() > 2) << "dimension exceeds 2";
    EigenMat mat(namedTensor_.second.flat<Dtype>().data(), dim_[0], dim_[1]);
    return mat.col(colId);
  }

  typename EigenMat::RowXpr row(int rowId) {
    LOG_IF(FATAL, dim_.size() > 2) << "dimension exceeds 2";
    EigenMat mat(namedTensor_.second.flat<Dtype>().data(), dim_[0], dim_[1]);
    return mat.row(rowId);
  }

  EigenMat eMat() {
    LOG_IF(FATAL, dim_.size() > 2) << "dimension exceeds 2";
    EigenMat mat(namedTensor_.second.flat<Dtype>().data(), dim_[0], dim_[1]);
    return mat;
  }

  //////////////////////////
  /// Eigen Tensor mirror///
  //////////////////////////
  EigenTensor eTensor() {
    return EigenTensor(namedTensor_.second.flat<Dtype>().data(), esizes_);
  }

  void setConstant(Dtype constant) {
    eTensor().setConstant(constant);
  }

  void setZero() {
    eTensor().setZero();
  }

  Dtype *data() {
    return eTensor().data();
  }

  ////////////////////////////////
  /// tensorflow tensor mirror ///
  ////////////////////////////////
  tensorflow::TensorShape tfShape() {
    return namedTensor_.second.shape();
  }

  const tensorflow::Tensor &tfTensor() const {
    return namedTensor_.second;
  }

  rai::Vector<tensorflow::Tensor> &output() {
    return vecTens;
  }

  ///////////////////////////////
  ////////// operators //////////
  ///////////////////////////////
  void operator=(const tensorflow::Tensor &tfTensor) {
    LOG_IF(FATAL, dim_inv_ != tfTensor.shape()) << "Tensor shape mismatch";
    std::memcpy(namedTensor_.second.flat<Dtype>().data(), tfTensor.flat<Dtype>().data(), sizeof(Dtype) * size_);
  }

  void operator=(const std::string name) {
    setName(name);
  }

  void operator=(const Eigen::Tensor<Dtype, NDim> &eTensor) {
    for (int i = 0; i < NDim; i++)
      LOG_IF(FATAL, dim_[i] != eTensor.dimension(i))
      << "Tensor size mismatch: " << i << "th Dim " << dim_[i] << "vs" << eTensor.dimension(i);
    std::memcpy(namedTensor_.second.flat<Dtype>().data(), eTensor.data(), sizeof(Dtype) * size_);
  }

  template<int rows, int cols>
  void operator=(const Eigen::Matrix<Dtype, rows, cols> &eMat) {
    LOG_IF(FATAL, dim_.size() > 2) << "The dimension is higher than 2";
    LOG_IF(FATAL, dim_[0] != eMat.rows() || dim_[1] != eMat.cols())
    << "matrix size mismatch: " << dim_[0] << "X" << dim_[1] << "vs" << eMat.rows() << "X"
    << eMat.cols();
    std::memcpy(namedTensor_.second.flat<Dtype>().data(), eMat.data(), sizeof(Dtype) * size_);
  }

  Dtype *operator[](int x) {
    return vecTens[x].flat<Dtype>().data();
  };

  ///////////////////////////////
  /////////// generic ///////////
  ///////////////////////////////
  const std::string &getName() const { return namedTensor_.first; }
  void setName(const std::string name) { namedTensor_.first = name; }
  const rai::Vector<int> &dim() const { return dim_; }
  const int dim(int idx) const { return dim_[idx]; }
  int rows() { return dim_[0]; }
  int cols() { return dim_[1]; }
  int batches() { return dim_[2]; }

  /// you lose all data calling resize
  void resize(const rai::Vector<int> dim) {
    LOG_IF(FATAL, NDim != dim.size()) << "tensor rank mismatch";
    dim_inv_.Clear();
    size_ = 1;
    dim_ = dim;
    for (int i = dim.size() - 1; i > -1; i--) {
      dim_inv_.AddDim(dim[i]);
      size_ *= dim[i];
    }
    for (int d = 0; d < NDim; d++)
      esizes_[d] = dim_[d];
    setDataType();
    namedTensor_.second = tensorflow::Tensor(dtype_, dim_inv_);
  }

  /// you lose all data calling resize
  void resize(int rows, int cols) {
    rai::Vector<int> dim = {rows, cols};
    resize(dim);
  }

  /// you lose all data calling resize
  void resize(int rows, int cols, int batches) {
    rai::Vector<int> dim = {rows, cols, batches};
    resize(dim);
  }

  /////////// 3D methods /////////
  typename EigenMat::ColXpr col(int batchId, int colId) {
    LOG_IF(FATAL, dim_.size() != 3) << "This is 3D Tensor method";
    EigenMat mat(namedTensor_.second.flat<Dtype>().data() + batchId * dim_[0] * dim_[1], dim_[0], dim_[1]);
    return mat.col(colId);
  }

  typename EigenMat::RowXpr row(int batchId, int rowId) {
    LOG_IF(FATAL, dim_.size() != 3) << "This is 3D Tensor method";
    EigenMat mat(namedTensor_.second.flat<Dtype>().data() + batchId * dim_[0] * dim_[1], dim_[0], dim_[1]);
    return mat.row(rowId);
  }

  EigenMat batch(int batchId) {
    LOG_IF(FATAL, dim_.size() != 3) << "This is 3D Tensor method";
    EigenMat mat(namedTensor_.second.flat<Dtype>().data() + batchId * dim_[0] * dim_[1], dim_[0], dim_[1]);
    return mat;
  }

  template<int rows, int cols>
  void partiallyFillBatch(int batchId, Eigen::Matrix<Dtype, rows, cols> &eMat) {
    LOG_IF(FATAL, dim_.size() != 3) << "This is 3D Tensor method";
    LOG_IF(FATAL, dim_[0] != rows) << "Column size mismatch ";
    std::memcpy(namedTensor_.second.flat<Dtype>().data() + batchId * dim_[0] * dim_[1],
                eMat.data(), sizeof(Dtype) * eMat.size());
  }

  template<int rows>
  void partiallyFillBatch(int batchId, rai::Vector<Eigen::Matrix<Dtype, rows, 1>> &eMatVec, int ignoreLastN = 0) {
    LOG_IF(FATAL, dim_.size() != 3) << "This is 3D Tensor method";
    LOG_IF(FATAL, dim_[0] != rows) << "Column size mismatch ";
    for (int colId = 0; colId < eMatVec.size() - ignoreLastN; colId++)
      batch(batchId).col(colId) = eMatVec[colId];
  }

 private:

  void init(const rai::Vector<int> dim, const std::string name = "") {
    LOG_IF(FATAL, dim.size() != NDim)
    << "specified dimension differs from the Dimension in the template parameter";
    namedTensor_.first = name;
    setDataType();
    size_ = 1;
    dim_ = dim;
    for (int i = dim_.size() - 1; i > -1; i--) {
      dim_inv_.AddDim(dim_[i]);
      size_ *= dim_[i];
    }
    namedTensor_ = {name, tensorflow::Tensor(dtype_, dim_inv_)};
    for (int d = 0; d < NDim; d++)
      esizes_[d] = dim_[d];
  }

  void setDataType() {
    if (typeid(Dtype) == typeid(float))
      dtype_ = tensorflow::DataType::DT_FLOAT;
    else if (typeid(Dtype) == typeid(double))
      dtype_ = tensorflow::DataType::DT_DOUBLE;
  }

  tensorflow::DataType dtype_;
  std::pair<std::string, tensorflow::Tensor> namedTensor_;
  rai::Vector<int> dim_;
  tensorflow::TensorShape dim_inv_; /// tensorflow dimension
  long int size_;
  Eigen::DSizes<Eigen::DenseIndex, NDim> esizes_;
  rai::Vector<tensorflow::Tensor> vecTens;

};

template<typename Dtype, int NDim>
std::ostream &operator<<(std::ostream &os, Tensor<Dtype, NDim> &m) {
  os << m.eTensor();
  return os;
}

}

#endif //RAI_RAI_TENSOR_HPP
