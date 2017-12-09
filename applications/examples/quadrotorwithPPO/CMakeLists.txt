add_subdirectory(${RAI_ROOT}/RAI/src/tasks/quadrotor ${CMAKE_BINARY_DIR}/applications/examples/quadrotorwithPPO/src)

add_executable(quadrotor_PPO
        ${RAI_TASK_SRC}
        quadrotor_PPO.cpp)

target_include_directories(quadrotor_PPO PUBLIC)
target_link_libraries(quadrotor_PPO ${RAI_LINK})