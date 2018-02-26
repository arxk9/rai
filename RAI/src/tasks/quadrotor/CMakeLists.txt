set(RAI_TASK_SRC
        ${RAI_TASK_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/visualizer/Quadrotor_Visualizer.cpp )
set(RAI_LINK ${RAI_LINK} glut GLU GL GLEW tinyxml ${RAI_Graphics} PARENT_SCOPE)
set(RAI_TASK_SRC ${RAI_TASK_SRC} PARENT_SCOPE)