cmake_minimum_required(VERSION 3.5)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_AUTOMOC ON)
add_definitions(-std=c++11 -pthread)

#gcc
set(CMAKE_CXX_COMPILER "/usr/bin/g++-6")
set(CMAKE_CXX_FLAGS "-Wl,--no-as-needed -fopenmp -O3 -w -funroll-loops")
link_directories(/usr/local/lib)
link_directories(/usr/lib)
link_directories(/usr/lib/x86_64-linux-gnu/)

project(RAI)
set(RAI_ROOT ${CMAKE_CURRENT_SOURCE_DIR})


###############################
########## Packages ###########
###############################
find_package(Boost 1.54.0 COMPONENTS filesystem system program_options date_time REQUIRED)

#OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

################################
######## build options #########
################################
option(VIS_ON "Toggle the visualizer." OFF)
option(BUILD_TEST "building test files" OFF)

################################
####### Common Libraries #######
################################
set(RAI_Link ${RAI_Link} boost_system boost_filesystem boost_iostreams glog gflags)

################################
#### function approximators ####
################################
include("${PROJECT_SOURCE_DIR}/cmake/tensorflowDependencies/tensorflow.cmake")
include_directories(${TENSORFLOW_INCLUDE_DIRS})
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tensorflowDependencies" ${CMAKE_MODULE_PATH})
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deepLearning/tensorflowWrapper/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deepLearning/tensorflow/bazel-genfiles/)
set(RAI_Link ${RAI_Link} ${TENSORFLOW_LIBRARIES} ${TENSORFLOW_SHARED_LIBRARY})
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tensorflowDependencies" ${CMAKE_MODULE_PATH})

################################
##### Include directories ######
################################
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/RAI/include)
include_directories(utils)
include_directories(common)
include_directories(graphics)

################################
####### sub directories ########
################################
add_subdirectory(utils)
set(RAI_Graphics)
add_subdirectory(graphics)
if (BUILD_TEST)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/applications/tests)
endif ()

################################
########## Application #########
################################
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/applications/${RAI_APP})
