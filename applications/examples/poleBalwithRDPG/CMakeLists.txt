add_subdirectory(${RAI_ROOT}/RAI/src/tasks/poleBalancing ${CMAKE_BINARY_DIR}/applications/examples/poleBalwithRDPG/src)

# 1D Pole Balancing Task
add_executable(poleBal_RDPG
        ${RAI_TASK_SRC}
        poleBalwithRDPG.cpp)

target_include_directories(poleBal_RDPG PUBLIC)
target_link_libraries(poleBal_RDPG ${RAI_LINK})
