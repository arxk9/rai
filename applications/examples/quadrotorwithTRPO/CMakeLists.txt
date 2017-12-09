add_subdirectory(${RAI_ROOT}/RAI/src/tasks/quadrotor ${CMAKE_BINARY_DIR}/applications/examples/quadrotorwithTRPO/src)

add_executable(quadrotor_TRPO
        ${RAI_TASK_SRC}
        quadrotor_TRPO.cpp)

target_include_directories(quadrotor_TRPO PUBLIC)
target_link_libraries(quadrotor_TRPO ${RAI_LINK})