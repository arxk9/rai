add_subdirectory(RAI_graph)
add_subdirectory(RAI_logger)
add_subdirectory(RAI_timer)

set(RAI_Link ${RAI_Link} rai_graph rai_timer rai_logger PARENT_SCOPE)