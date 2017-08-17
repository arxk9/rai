cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-w")

set(NAME_OF_TENSORFLOW_SRC_FOLDER tensorflow_source)
set(PATH_TO_TENSORFLOW_ROOT ../../tensorflow/${NAME_OF_TENSORFLOW_SRC_FOLDER})
set(PATH_TO_TENSORFLOW_CMAKE_BUILD ../../tensorflow/tensorflow_cmake_build)

get_filename_component(ABSOLUTE_PATH_TO_TENSORFLOW_ROOT ${PATH_TO_TENSORFLOW_ROOT} REALPATH)
get_filename_component(ABSOLUTE_PATH_TO_TENSORFLOW_CMAKE_BUILD ${PATH_TO_TENSORFLOW_CMAKE_BUILD} REALPATH)

add_definitions(-std=c++11)

include_directories(../include)

add_executable(testTensorFlowNeuralNetwork testTensorFlowNeuralNetwork.cpp)

include_directories(${ABSOLUTE_PATH_TO_TENSORFLOW_ROOT}/bazel-${NAME_OF_TENSORFLOW_SRC_FOLDER})
include_directories(${PATH_TO_TENSORFLOW_CMAKE_BUILD}) # For the *.pb.h and *.pb.cc files
include_directories(${ABSOLUTE_PATH_TO_TENSORFLOW_ROOT}/bazel-${NAME_OF_TENSORFLOW_SRC_FOLDER}/external/eigen_archive/eigen-eigen-b4fa9622b809)
include_directories(${ABSOLUTE_PATH_TO_TENSORFLOW_ROOT}/bazel-${NAME_OF_TENSORFLOW_SRC_FOLDER}/external/eigen_archive)
target_link_libraries(testTensorFlowNeuralNetwork ${ABSOLUTE_PATH_TO_TENSORFLOW_ROOT}/bazel-bin/tensorflow/libtensorflow.so)
target_link_libraries(testTensorFlowNeuralNetwork /usr/local/cuda/lib64/libcudart.so)
target_link_libraries(testTensorFlowNeuralNetwork glog)

file(COPY resources DESTINATION ${CMAKE_BINARY_DIR})
