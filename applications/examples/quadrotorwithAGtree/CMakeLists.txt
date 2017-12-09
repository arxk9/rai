add_subdirectory(${RAI_ROOT}/RAI/src/tasks/quadrotor ${CMAKE_BINARY_DIR}/applications/examples/quadrotorwithAGtree/src)

add_executable(quadrotor_AGtree
        ${RAI_TASK_SRC}
        quadrotor_AGtree.cpp)

target_include_directories(quadrotor_AGtree PUBLIC)
target_link_libraries(quadrotor_AGtree ${RAI_LINK})

add_executable(quadrotor_sim
        ${RAI_TASK_SRC}
        simulator.cpp)

target_include_directories(quadrotor_sim PUBLIC)
target_link_libraries(quadrotor_sim ${RAI_LINK})
