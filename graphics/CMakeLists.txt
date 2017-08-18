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
link_directories(/usr/lib)

INCLUDE(FindPkgConfig)

add_library(rai_graphics ${SRCfiles})
set(RAI_Graphics rai_graphics glut GLU GL GLEW SDL2 assimp SOIL glog freeimage pthread)
add_executable(testRAIGraphics test.cpp)
target_link_libraries(testRAIGraphics ${RAI_Graphics})

install(DIRECTORY obj imp DESTINATION ${CMAKE_BINARY_DIR}/include)
install(FILES RAI_graphics.hpp RAI_keyboard.hpp DESTINATION ${CMAKE_BINARY_DIR}/include)
install(TARGETS rai_graphics EXPORT RAITargets DESTINATION ${CMAKE_BINARY_DIR}/lib)
set(RAI_LINK ${RAI_Graphics} ${RAI_LINK} PARENT_SCOPE)
