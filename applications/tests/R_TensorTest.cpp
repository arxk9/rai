//
// Created by jhwangbo on 12/08/17.
//

#include "rai/RAI_Tensor.hpp"

void test(Eigen::Matrix<float, -1, -1> ten2D);
void test2(std::pair<std::string, tensorflow::Tensor> ten);

int main() {

  rai::Tensor<float, 2> ten({3, 4}, "test tensor");
  Eigen::Matrix<float, 3, 4> eMat;
  Eigen::Tensor<float, 2> eigenTen(3, 4);
  tensorflow::Tensor tfTensor(tensorflow::DataType::DT_FLOAT, tensorflow::TensorShape({4, 3}));
  ten.row(1);
  eMat.setConstant(1);
  eigenTen.setConstant(2);
  tfTensor.tensor<float, 2>().setConstant(3);
  /// testing '=' operator
  ten.setZero();
  std::cout << "my tensor (should be 0) " << std::endl << ten << std::endl;
  ten = eMat;
  ten = "ten";
  std::cout << "my tensor (should be 1)" << std::endl << ten << std::endl;
  ten = eigenTen;
  std::cout << "my tensor (should be 2)" << std::endl << ten << std::endl;
  ten = tfTensor;
  std::cout << "my tensor (should be 3)" << std::endl << ten << std::endl;

  /// checking 2d Eigen Matrix operation test
  Eigen::Vector3f eigenVec(7, 7, 7);
  Eigen::RowVector4f eigenVec2(4, 4, 4, 4);
  ten.col(0) = eigenVec;
  ten.row(2) = eigenVec2;
  std::cout << "my tensor (first column should be 7, 3rd row is 4)" << std::endl << ten << std::endl;

  std::cout << "ten block(0,0,2,1)" << std::endl << ten.block(0, 0, 2, 1) << std::endl;
  std::cout << "ten block(1,2,2,2)" << std::endl << ten.block(1, 2, 2, 2) << std::endl;

  /// checking 3d methods
  rai::Tensor<float, 3> ten3D({3, 4, 4}, "testTensor");
  ten3D.setConstant(1.5);
  ten3D.batch(2) = eMat;
  std::cout << "3rd batch of tensor should be 1" << std::endl << ten3D << std::endl;

  ten3D.partiallyFillBatch(1, eigenVec);
  std::cout << "1st column fo the 2rd batch of tensor should be 7" << std::endl << ten3D << std::endl;

  ten3D.resize({3, 4, 2});
  std::cout << "there should be 24 numbers" << std::endl << ten3D << std::endl;

  ten3D.setZero();
  for (int i = 0; i < ten3D.batches(); i++) {
    ten3D.col(i, 1) << 1 + 3 * i, 2 + 3 * i, 3 + 3 * i;
  }
  std::cout << "2nd columns of each batches are nonzero" << std::endl << ten3D << std::endl;
  std::cout << "all 2nd rows(shape = {4,2}) [ten3D.row(1)]" << std::endl << ten3D.row(1) << std::endl;
  std::cout << "all 2nd columns(shape = {3,2}) [ten3D.col(1)]" << std::endl << ten3D.col(1) << std::endl;

  rai::Tensor<float, 1> ten1D("testTensor");
  ten1D.resize(4);

  for (int i = 0; i < ten1D.dim(0); i++)
    ten1D[i] = i;
  std::cout << "1D tensor" << std::endl << ten1D << std::endl;
  test(ten);
  test2(ten);

}

void test(Eigen::Matrix<float, -1, -1> ten2D) {
  std::cout << "casting to eigenMat" << std::endl << ten2D << std::endl;
}

void test2(std::pair<std::string, tensorflow::Tensor> ten) {
  std::cout << "casting to tfTensor" << std::endl << ten.first << std::endl;
}
