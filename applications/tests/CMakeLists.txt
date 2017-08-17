add_executable(randomSampling testRandomSampling.cpp)
target_link_libraries(randomSampling rai_graph glog)

add_executable(conjugateGradient testConjugateGradient.cpp)
target_link_libraries(conjugateGradient ${RAI_Link})

add_executable(RTensor R_TensorTest.cpp)
target_link_libraries(RTensor ${RAI_Link})

add_subdirectory(${RAI_ROOT}/RAI/include/rai/tasks/hopperBox2D/tests ${CMAKE_CURRENT_BINARY_DIR}/tests/hopperBox2D)
#add_subdirectory(${RAI_ROOT}/RAI/include/rai/tasks/poleBalancing/tests ${CMAKE_CURRENT_BINARY_DIR}/tests/poleBalancing)
add_subdirectory(${RAI_ROOT}/RAI/include/rai/algorithm/tests ${CMAKE_CURRENT_BINARY_DIR}/tests/algorithm)
add_subdirectory(${RAI_ROOT}/RAI/include/rai/tasks/poleBalancing1D/tests ${CMAKE_CURRENT_BINARY_DIR}/tests/poleBalancing1D)
add_subdirectory(${RAI_ROOT}/RAI/include/rai/noiseModel/tests ${CMAKE_CURRENT_BINARY_DIR}/tests/noiseModel)

add_executable(testTensorFlow testTensorFlow.cpp)
target_link_libraries(testTensorFlow ${RAI_Link})

add_subdirectory(${RAI_ROOT}/RAI/include/rai/function/tensorflow/pythonProtobufGenerators/test ${CMAKE_BINARY_DIR}/RAI/include/functionApproximator/tensorflow/pythonProtobufGenerators/test)
