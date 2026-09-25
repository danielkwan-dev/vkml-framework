add_executable(vkml-info vkml_info.cpp)
target_link_libraries(vkml-info PRIVATE vkml::vkml)
vkml_set_warnings(vkml-info)

add_executable(vkml-run vkml_run.cpp)
target_link_libraries(vkml-run PRIVATE vkml::vkml)
vkml_set_warnings(vkml-run)

add_executable(vkml-bench vkml_bench.cpp)
target_link_libraries(vkml-bench PRIVATE vkml::vkml)
vkml_set_warnings(vkml-bench)

add_executable(vkml-tokenize vkml_tokenize.cpp)
target_link_libraries(vkml-tokenize PRIVATE vkml::vkml nlohmann_json::nlohmann_json)
vkml_set_warnings(vkml-tokenize)
