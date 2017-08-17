find_package(SFML 2 REQUIRED system window graphics network audio)
include_directories(${SFML_INCLUDE_DIR})
set(RAI_Link ${SFML_LIBRARIES} ${RAI_Link} PARENT_SCOPE)
