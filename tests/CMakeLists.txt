include(FetchContent)
FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.16.0
    GIT_SHALLOW    TRUE
    SYSTEM)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
include(Catch)

add_executable(vkml_tests
    test_context.cpp
    test_activations.cpp
    test_buffer.cpp
    test_embedding.cpp
    test_matmul.cpp
    test_norms.cpp
    test_ops.cpp
    test_pipeline.cpp
    test_stream.cpp
    test_tensor.cpp
)
# HAL tests reach into src/ directly, so they need the same private deps as vkml.
target_include_directories(vkml_tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(vkml_tests PRIVATE vkml::vkml vkml_hal_deps vkml_shaders Catch2::Catch2WithMain)
vkml_set_warnings(vkml_tests)

catch_discover_tests(vkml_tests)
