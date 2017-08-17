add_subdirectory(${RAI_ROOT}/RAI/src/tasks/poleBalancing ${CMAKE_BINARY_DIR}/applications/examples/poleBalwithDDPG/src)

# 1D Pole Balancing Task
add_executable(poleBal_DDPG
        ${RAI_TASK_SRC}
        poleBalwithDDPG.cpp)
add_executable(poleBal_DDPG_simp
        ${RAI_TASK_SRC}
        simplifiedVersion.cpp)

target_include_directories(poleBal_DDPG PUBLIC)
target_link_libraries(poleBal_DDPG ${RAI_Link})

target_include_directories(poleBal_DDPG_simp PUBLIC ${SFML_INCLUDE_DIR})
target_link_libraries(poleBal_DDPG_simp ${SFML_LIBRARIES} ${RAI_Link})
