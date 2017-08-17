# 1D Pole Balancing Task
add_subdirectory(${RAI_ROOT}/RAI/src/tasks/poleBalancing ${CMAKE_BINARY_DIR}/applications/examples/poleBalwithTRPO/src)

add_executable(poleBal_TRPO
        ${RAI_TASK_SRC}
        poleBalwithTRPO.cpp)
target_include_directories(poleBal_TRPO PUBLIC)
target_link_libraries(poleBal_TRPO ${RAI_Link})
