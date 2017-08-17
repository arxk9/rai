cmake_minimum_required(VERSION 3.5)

# Use C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Include header folders
include_directories( ${Eigen3_INCLUDE_DIRS} )
include_directories(include)

# Export the plotter library
add_library(rai_timer RAI_timer.cpp)

install(TARGETS rai_timer
        # IMPORTANT: Add the bar executable to the "export-set"
        EXPORT RAITargets
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin)