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
include(CheckCXXCompilerFlag)
include(GNUInstallDirs)

project(RAI)

##############################
############ Path ############
##############################
set(CMAKEPACKAGE_INSTALL_DIR
        "/${CMAKE_INSTALL_DATADIR}/RAI/cmake"
        CACHE PATH "The directory relative to CMAKE_PREFIX_PATH where RAIConfig.cmake is installed"
        )

set(PKGCONFIG_INSTALL_DIR
        "/${CMAKE_INSTALL_DATADIR}/pkgconfig"
        CACHE PATH "The directory relative to CMAKE_PREFIX_PATH where RAI.pc is installed"
        )

set(RAI_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

###############################
########## Packages ###########
###############################
#OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

################################
######## build options #########
################################
option(BUILD_TEST "building test files" OFF)

################################
####### Common Libraries #######
################################
set(RAI_LINK glog gflags CACHE INTERNAL "")

################################
#### function approximators ####
################################
include("${PROJECT_SOURCE_DIR}/cmake/tensorflowDependencies/tensorflow.cmake")
include_directories(${TENSORFLOW_INCLUDE_DIRS})
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tensorflowDependencies" ${CMAKE_MODULE_PATH})
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deepLearning/tensorflow/bazel-genfiles/)
set(RAI_LINK ${RAI_LINK} ${TENSORFLOW_LIBRARIES} ${TENSORFLOW_SHARED_LIBRARY})
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tensorflowDependencies" ${CMAKE_MODULE_PATH})

################################
##### Include directories ######
################################
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/RAI/include)
include_directories(utils)
include_directories(common)
include_directories(graphics)
set(RAI_LINK rai_logger rai_timer rai_graph ${RAI_LINK})

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
if (NOT ${RAI_APP} STREQUAL "")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/applications/${RAI_APP})
endif ()

################################
######## package config ########
################################
install(EXPORT RAITargets DESTINATION ${CMAKE_BINARY_DIR})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/RAI/include DESTINATION ${CMAKE_BINARY_DIR})
include(CMakePackageConfigHelpers)

set(RAI_INCLUDE_DIR ${CMAKE_BINARY_DIR}/include CACHE PATH "rai include directory")
set(RAI_INCLUDE_DIR ${RAI_INCLUDE_DIR} ${TENSORFLOW_INCLUDE_DIRS})
message("RAI_INCLUDE_DIR ${RAI_INCLUDE_DIR}")
set(RAI_LIB_DIR ${CMAKE_BINARY_DIR}/lib CACHE PATH "rai lib directory")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RAIConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/RAIConfig.cmake @ONLY
        )
configure_file(RAI.pc.in RAI.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/RAI.pc
        DESTINATION ${CMAKE_INSTALL_PREFIX}${PKGCONFIG_INSTALL_DIR}
        )