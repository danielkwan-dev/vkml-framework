add_executable(vkml-info vkml_info.cpp)
target_link_libraries(vkml-info PRIVATE vkml::vkml)
vkml_set_warnings(vkml-info)
