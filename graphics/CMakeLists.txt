cmake_minimum_required(VERSION 2.8)
add_definitions(-std=c++11 -pthread)
set(CMAKE_AUTOMOC ON)

file(GLOB SRCfiles
        "RAI_graphics.hpp"
        "RAI_graphics.cpp"
        "./imp/*.cpp"
        "./obj/*.cpp"
        )

find_package( Eigen3 REQUIRED )
include_directories( ${EIGEN3_INCLUDE_DIR} )
add_executable(testRAIGraphics test.cpp ${SRCfiles})
link_directories(/usr/lib)

INCLUDE(FindPkgConfig)
message("rai link is graph is ${RAI_Link}")
target_link_libraries(testRAIGraphics glut GLU GL GLEW SDL2 assimp SOIL glog freeimage ${RAI_Link} pthread ${RAI_TARGET})

add_library(rai_graphics ${SRCfiles})
set(RAI_Graphics ${RAI_Graphics} rai_graphics glut GLU GL GLEW SDL2 assimp SOIL glog freeimage pthread PARENT_SCOPE)

