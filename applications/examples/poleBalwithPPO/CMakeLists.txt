# 1D Pole Balancing Task
add_subdirectory(${RAI_ROOT}/RAI/src/tasks/poleBalancing ${CMAKE_BINARY_DIR}/applications/examples/poleBalwithPPO/src)

add_executable(poleBal_PPO ${RAI_TASK_SRC} poleBalwithPPO.cpp)
target_include_directories(poleBal_PPO PUBLIC)
target_link_libraries(poleBal_PPO ${RAI_LINK})

add_executable(poleBal_PPO_RP ${RAI_TASK_SRC} poleBalwithPPO_recurrentPolicy.cpp)
target_include_directories(poleBal_PPO_RP PUBLIC)
target_link_libraries(poleBal_PPO_RP ${RAI_LINK})
