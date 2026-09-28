cmake_minimum_required(VERSION 3.25)

project(vkml
    VERSION 0.1.0
    DESCRIPTION "GPU tensor library and LLM inference engine on Vulkan compute"
    HOMEPAGE_URL "https://github.com/danielkwan-dev/vkml-framework"
    LANGUAGES C CXX) # C is for volk.c

option(VKML_BUILD_TESTS "Build the test suite" ${PROJECT_IS_TOP_LEVEL})
option(VKML_BUILD_APPS "Build the command-line tools" ${PROJECT_IS_TOP_LEVEL})
option(VKML_BUILD_EXAMPLES "Build the examples" ${PROJECT_IS_TOP_LEVEL})
option(VKML_WERROR "Treat compiler warnings as errors" OFF)

set(CMAKE_CXX_EXTENSIONS OFF)

# MinGW programs otherwise need libstdc++, libgcc and winpthread DLLs at run
# time, and Windows may load incompatible copies from another toolchain on PATH
# (Git for Windows ships some), failing every test with an error dialog.
if(MINGW)
    add_link_options(-static)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(Warnings)
include(Dependencies)
include(Shaders)

vkml_add_shaders(vkml_shaders
    shaders/binary.comp
    shaders/cast.comp
    shaders/dequantize.comp
    shaders/embedding.comp
    shaders/fill.comp
    shaders/gemv.comp
    shaders/gemv_dot.comp
    shaders/matmul.comp
    shaders/permute.comp
    shaders/quantize.comp
    shaders/rms_norm.comp
    shaders/rope.comp
    shaders/softmax.comp
    shaders/unary.comp
    shaders/write_rows.comp
)

# Static on purpose: volk keeps the Vulkan dispatch table in global variables,
# and a shared vkml would give the library and its tests separate copies.
add_library(vkml STATIC
    src/core/context.cpp
    src/core/runtime.cpp
    src/core/tensor.cpp
    src/io/chat.cpp
    src/io/jinja.cpp
    src/io/pretokenize.cpp
    src/io/safetensors.cpp
    src/io/sampling.cpp
    src/io/tokenizer.cpp
    src/io/unicode.cpp
    src/models/llama.cpp
    src/ops/attention.cpp
    src/ops/cast.cpp
    src/ops/elementwise.cpp
    src/ops/embedding.cpp
    src/ops/matmul.cpp
    src/ops/norms.cpp
    src/ops/permute.cpp
    src/ops/quantize.cpp
    src/ops/rope.cpp
    src/hal/buffer.cpp
    src/hal/device.cpp
    src/hal/pipeline.cpp
    src/hal/stream.cpp
    src/hal/vma.cpp
)
add_library(vkml::vkml ALIAS vkml)

target_include_directories(vkml
    PUBLIC  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    PRIVATE ${PROJECT_SOURCE_DIR}/src
)
target_compile_features(vkml PUBLIC cxx_std_20)
target_link_libraries(vkml PRIVATE vkml_hal_deps vkml_shaders nlohmann_json::nlohmann_json)
vkml_set_warnings(vkml)

if(VKML_BUILD_APPS)
    add_subdirectory(apps)
endif()

if(VKML_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(VKML_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
