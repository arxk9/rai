cmake_minimum_required(VERSION 3.5)

project(gnuplotWrapper)

# Use C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Include header folders
include_directories( ${Eigen3_INCLUDE_DIRS} )
include_directories(include)

# Export the plotter library
add_library(rai_graph src/gnuplotter.cpp)

# Build usage examples
add_executable(plotting_examples src/plotting_examples.cpp)
target_link_libraries(plotting_examples rai_graph ${RAI_LINK})

install(DIRECTORY include DESTINATION ${CMAKE_BINARY_DIR}/include/RAI_graph)
install(TARGETS rai_graph EXPORT RAITargets DESTINATION ${CMAKE_BINARY_DIR}/lib)