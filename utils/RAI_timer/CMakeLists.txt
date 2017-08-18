cmake_minimum_required(VERSION 3.5)

# Use C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Include header folders
include_directories( ${Eigen3_INCLUDE_DIRS} )
include_directories(include)

# Export the plotter library
add_library(rai_timer RAI_timer.cpp)

install(FILES RAI_timer_ToInclude.hpp RAI_timer.hpp RAI_timer_items.hpp DESTINATION ${CMAKE_BINARY_DIR}/include/RAI_timer)
install(TARGETS rai_timer EXPORT RAITargets DESTINATION ${CMAKE_BINARY_DIR}/lib)