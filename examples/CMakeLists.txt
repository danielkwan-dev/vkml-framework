add_executable(quickstart quickstart.cpp)
target_link_libraries(quickstart PRIVATE vkml::vkml)
vkml_set_warnings(quickstart)
