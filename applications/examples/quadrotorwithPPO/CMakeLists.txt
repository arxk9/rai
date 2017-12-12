add_subdirectory(${RAI_ROOT}/RAI/src/tasks/quadrotor ${CMAKE_BINARY_DIR}/applications/examples/quadrotorwithPPO/src/quardotor)
add_subdirectory(${RAI_ROOT}/RAI/src/tasks/slungload ${CMAKE_BINARY_DIR}/applications/examples/quadrotorwithPPO/src/slungload)

add_executable(quadrotor_PPO
        ${RAI_TASK_SRC}
        quadrotor_PPO.cpp)

target_include_directories(quadrotor_PPO PUBLIC)
target_link_libraries(quadrotor_PPO ${RAI_LINK})

add_executable(quadrotor_RPPO
        ${RAI_TASK_SRC}
        quadrotorwithRPPO.cpp)

target_include_directories(quadrotor_RPPO PUBLIC)
target_link_libraries(quadrotor_RPPO ${RAI_LINK})

add_executable(slungload_RPPO
        ${RAI_TASK_SRC}
        slungloadwithRPPO.cpp)

target_include_directories(slungload_RPPO PUBLIC)
target_link_libraries(slungload_RPPO ${RAI_LINK})