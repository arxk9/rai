set(RAI_Link  ${RAI_Link} PARENT_SCOPE)

set(RAI_TASK_SRC
        ${RAI_TASK_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/visualizer/Pole_Visualizer.cpp )

set(RAI_Link ${RAI_Link} glut GLU GL GLEW urdfdom_model tinyxml ${RAI_Graphics})
set(RAI_Link ${RAI_Link} PARENT_SCOPE)
set(RAI_TASK_SRC ${RAI_TASK_SRC} PARENT_SCOPE)
