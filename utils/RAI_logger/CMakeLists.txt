cmake_minimum_required(VERSION 3.5)

# Use C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Include header folders
include_directories(include)

# Export the plotter library
add_library(rai_logger src/RAI_logger.cpp)

# Build usage examples
if(BUILD_TEST)
  add_executable(logging_examples src/logging_examples.cpp)
  target_link_libraries(logging_examples rai_logger glog)
endif()