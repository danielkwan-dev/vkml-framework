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

add_executable(vkml-chat vkml_chat.cpp)
target_link_libraries(vkml-chat PRIVATE vkml::vkml nlohmann_json::nlohmann_json)
vkml_set_warnings(vkml-chat)

# cpp-httplib, vkml-server's HTTP server. Only its header is used, without TLS
# or compression: its own CMake project would go looking for OpenSSL and zlib.
include(FetchContent)
FetchContent_Declare(httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.58.0
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  header-only)
FetchContent_MakeAvailable(httplib)
find_package(Threads REQUIRED)

add_executable(vkml-server vkml_server.cpp)
target_include_directories(vkml-server SYSTEM PRIVATE "${httplib_SOURCE_DIR}")
target_link_libraries(vkml-server PRIVATE vkml::vkml nlohmann_json::nlohmann_json Threads::Threads
                                          $<$<PLATFORM_ID:Windows>:ws2_32>)
vkml_set_warnings(vkml-server)

# cmake --install puts the tools in <prefix>/bin. They need nothing else at run
# time: the shaders are compiled in, and MinGW builds link statically.
include(GNUInstallDirs)
install(TARGETS vkml-info vkml-run vkml-bench vkml-tokenize vkml-chat vkml-server
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
