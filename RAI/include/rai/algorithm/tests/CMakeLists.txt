add_definitions(-std=c++11 -w -O3)


include("../../../../../cmake/tensorflowDependencies/tensorflow.cmake")
find_package(SFML 2 REQUIRED system window graphics network audio)

add_executable(testDeepDeterministicPolicyGradient testDeepDeterministicPolicyGradient.cpp)
target_include_directories(testDeepDeterministicPolicyGradient PUBLIC ${SFML_INCLUDE_DIR})
target_link_libraries(testDeepDeterministicPolicyGradient ${SFML_LIBRARIES})
use_tensorflow(testDeepDeterministicPolicyGradient)
target_link_libraries(testDeepDeterministicPolicyGradient graph boost_system boost_filesystem rai_logger glog)

file(COPY resources DESTINATION ${CMAKE_BINARY_DIR}/RAI/include/algorithm/tests/)
file(COPY ../../../../src/tasks/poleBalancing/visualizer DESTINATION ${CMAKE_BINARY_DIR}/RAI/include/algorithm/tests)
