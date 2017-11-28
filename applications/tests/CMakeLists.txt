add_executable(randomSampling testRandomSampling.cpp)
target_link_libraries(randomSampling rai_graph glog ${RAI_LINK})

add_executable(conjugateGradient testConjugateGradient.cpp)
target_link_libraries(conjugateGradient ${RAI_LINK})

add_executable(RTensor R_TensorTest.cpp)
target_link_libraries(RTensor ${RAI_LINK})

add_subdirectory(${RAI_ROOT}/RAI/include/rai/algorithm/tests ${CMAKE_CURRENT_BINARY_DIR}/tests/algorithm)
add_subdirectory(${RAI_ROOT}/RAI/include/rai/tasks/poleBalancing1D/tests ${CMAKE_CURRENT_BINARY_DIR}/tests/poleBalancing1D)
add_subdirectory(${RAI_ROOT}/RAI/include/rai/noiseModel/tests ${CMAKE_CURRENT_BINARY_DIR}/tests/noiseModel)
add_subdirectory(${RAI_ROOT}/RAI/src/tasks/poleBalancing ${CMAKE_BINARY_DIR}/applications/examples/poleBalwithPPO/src)

add_executable(testTensorFlow testTensorFlow.cpp)
target_link_libraries(testTensorFlow ${RAI_LINK})

add_subdirectory(${RAI_ROOT}/RAI/include/rai/function/tensorflow/pythonProtobufGenerators/test ${CMAKE_BINARY_DIR}/RAI/include/functionApproximator/tensorflow/pythonProtobufGenerators/test)

add_executable(testDeterministicmodel testDeterministicModel.cpp)
target_link_libraries(testDeterministicmodel ${RAI_LINK})

add_executable(testRnn ${RAI_TASK_SRC} test_RNN.cpp)
target_include_directories(testRnn PUBLIC)
target_link_libraries(testRnn ${RAI_LINK})

add_executable(memory ${RAI_TASK_SRC} testMemoryleak.cpp)
target_include_directories(memory PUBLIC)
target_link_libraries(memory ${RAI_LINK})

add_executable(data ${RAI_TASK_SRC} testData.cpp)
target_include_directories(data PUBLIC)
target_link_libraries(data ${RAI_LINK})
