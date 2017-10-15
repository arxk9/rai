//
// Created by jhwangbo on 12/08/17.
//

#include "rai/RAI_Tensor.hpp"

int main() {

  rai::Tensor<float, 2> ten({3, 2}, "test tensor");

  Eigen::Matrix<float, 3, 2> eMat;
  Eigen::Tensor<float, 2> eigenTen(3, 2);
  tensorflow::Tensor tfTensor(tensorflow::DataType::DT_FLOAT, tensorflow::TensorShape({2, 3}));

  eMat.setConstant(1);
  eigenTen.setConstant(2);
  tfTensor.tensor<float, 2>().setConstant(3);
  /// testing '=' operator
  ten.setZero();
  std::cout << "my tensor (should be 0) " << std::endl << ten << std::endl;
  ten = eMat;
  std::cout << "my tensor (should be 1)" << std::endl << ten << std::endl;
  ten = eigenTen;
  std::cout << "my tensor (should be 2)" << std::endl << ten << std::endl;
  ten = tfTensor;
  std::cout << "my tensor (should be 3)" << std::endl << ten << std::endl;

  /// checking 2d Eigen Matrix operation test
  Eigen::Vector3f eigenVec(7, 7, 7);
  Eigen::RowVector2f eigenVec2(4, 4);
  ten.col(0) = eigenVec;
  ten.row(2) = eigenVec2;
  std::cout << "my tensor (first column should be 7, 3rd row is 4)" << std::endl << ten << std::endl;

  /// checking 3d methods
  rai::Tensor<float, 3> ten3D({3, 2, 4}, "testTensor");
  ten3D.setConstant(1.5);
  ten3D.batch(2) = eMat;
  std::cout << "3rd batch of tensor should be 1" << std::endl << ten3D << std::endl;

  ten3D.partiallyFillBatch(1, eigenVec);
  std::cout << "1st column fo the 2rd batch of tensor should be 7" << std::endl << ten3D << std::endl;

  ten3D.resize({3, 2, 2});
  std::cout << "there should be 12 numbers" << std::endl << ten3D << std::endl;

}