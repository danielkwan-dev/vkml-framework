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
    test_cast.cpp
    test_context.cpp
    test_activations.cpp
    test_attention.cpp
    test_buffer.cpp
    test_embedding.cpp
    test_kv_cache.cpp
    test_llama.cpp
    test_matmul.cpp
    test_norms.cpp
    test_ops.cpp
    test_permute.cpp
    test_pretokenize.cpp
    test_pipeline.cpp
    test_rope.cpp
    test_safetensors.cpp
    test_stream.cpp
    test_tensor.cpp
    test_tokenizer.cpp
    test_unicode.cpp
)
# HAL tests reach into src/ directly, so they need the same private deps as vkml.
target_include_directories(vkml_tests PRIVATE "${PROJECT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(vkml_tests PRIVATE vkml::vkml vkml_hal_deps vkml_shaders Catch2::Catch2WithMain)
vkml_set_warnings(vkml_tests)

catch_discover_tests(vkml_tests)
