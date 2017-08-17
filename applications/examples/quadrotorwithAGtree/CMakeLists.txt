add_executable(quadrotor_AGtree
        ${RAI_TASKS_SRC}
        quadrotor_AGtree.cpp)
target_link_libraries(quadrotor_AGtree ${RAI_Link})

add_executable(quadrotor_sim
        ${RAI_TASKS_SRC}
        simulator.cpp)
target_link_libraries(quadrotor_sim ${RAI_Link})

file(COPY "${PROJECT_SOURCE_DIR}/RAI/taskModules/Quadrotor/TensorflowPB_Quad" DESTINATION ${CMAKE_BINARY_DIR}/applications/jemin/quadrotorwithAGtree)

