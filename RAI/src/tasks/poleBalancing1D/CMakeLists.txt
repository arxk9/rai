# Project configuration

#################
set(RAI_TASKS_INCL
        ${Boost_INCLUDE_DIRS}
        ${RAI_TASKS_INCL}
PARENT_SCOPE)

#################
set(RAI_TASKS_SRC
        ${RAI_TASKS_SRC}
PARENT_SCOPE)


#################
set(RAI_TASKS_LIBS
        ${RAI_TASKS_LIBS}
        ${Boost_LIBRARIES}
PARENT_SCOPE)

if(VIS_ON)
set(RAI_TASKS_SRC
    ${RAI_TASKS_SRC}
    ${CMAKE_CURRENT_SOURCE_DIR}/visualizer/visualizer.cpp
    PARENT_SCOPE)
set(RAI_TASKS_LIBS
    ${RAI_TASKS_LIBS}
    glut
    GLU
    GL
    GLEW
PARENT_SCOPE)
endif()