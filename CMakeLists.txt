cmake_minimum_required(VERSION 3.25)

project(vkml
    VERSION 0.1.0
    DESCRIPTION "GPU tensor library and LLM inference engine on Vulkan compute"
    HOMEPAGE_URL "https://github.com/danielkwan-dev/vkml-framework"
    LANGUAGES C CXX) # C is for volk.c

option(VKML_BUILD_TESTS "Build the test suite" ${PROJECT_IS_TOP_LEVEL})
option(VKML_BUILD_APPS "Build the command-line tools" ${PROJECT_IS_TOP_LEVEL})
option(VKML_WERROR "Treat compiler warnings as errors" OFF)

set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(Warnings)
include(Dependencies)
include(Shaders)

vkml_add_shaders(vkml_shaders
    shaders/fill.comp
)

# Static on purpose: volk keeps the Vulkan dispatch table in global variables,
# and a shared vkml would give the library and its tests separate copies.
add_library(vkml STATIC
    src/core/context.cpp
    src/hal/device.cpp
    src/hal/pipeline.cpp
    src/hal/vma.cpp
)
add_library(vkml::vkml ALIAS vkml)

target_include_directories(vkml
    PUBLIC  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    PRIVATE ${PROJECT_SOURCE_DIR}/src
)
target_compile_features(vkml PUBLIC cxx_std_20)
target_link_libraries(vkml PRIVATE vkml_hal_deps vkml_shaders)
vkml_set_warnings(vkml)

if(VKML_BUILD_APPS)
    add_subdirectory(apps)
endif()

if(VKML_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
