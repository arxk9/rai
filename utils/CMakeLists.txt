add_subdirectory(RAI_graph)
add_subdirectory(RAI_logger)
add_subdirectory(RAI_timer)

set(RAI_TARGET ${RAI_TARGET} rai_graph rai_timer rai_logger PARENT_SCOPE)