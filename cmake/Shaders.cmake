# Compiles GLSL compute shaders to SPIR-V at build time and embeds each one
# in a generated C++ header, so the library never reads shader files at runtime.
#
#   vkml_add_shaders(<target> shaders/foo.comp ...)
#
# creates an INTERFACE target <target>. Linking it makes
#   #include <vkml_shaders/foo.spv.hpp>   ->   vkml::shaders::foo  (std::uint32_t[])
# available.

find_program(VKML_GLSLC glslc
    HINTS "$ENV{VULKAN_SDK}/bin" "$ENV{VULKAN_SDK}/Bin"
    REQUIRED
    DOC "glslc from shaderc (Vulkan SDK, or the glslc / shaderc system package)")

set(VKML_EMBED_SPIRV_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/EmbedSpirv.cmake")

# Whether glslc knows GL_EXT_integer_dot_product, which the int8 dot-product
# matrix-vector kernel needs. Older shaderc (2023.8, as Ubuntu 24.04 ships)
# does not; vkml then builds without that kernel.
if(NOT DEFINED VKML_GLSLC_INTEGER_DOT)
    set(probe "${CMAKE_BINARY_DIR}/glslc_probe/integer_dot.comp")
    file(WRITE "${probe}"
        "#version 450\n#extension GL_EXT_integer_dot_product : require\n"
        "layout(local_size_x = 1) in;\n"
        "layout(std430, binding = 0) buffer B { int v[]; };\n"
        "void main() { v[0] = dotPacked4x8EXT(v[1], v[2]); }\n")
    execute_process(
        COMMAND "${VKML_GLSLC}" --target-env=vulkan1.2 -o "${probe}.spv" "${probe}"
        RESULT_VARIABLE probe_result OUTPUT_QUIET ERROR_QUIET)
    if(probe_result EQUAL 0)
        set(supported ON)
    else()
        set(supported OFF)
    endif()
    set(VKML_GLSLC_INTEGER_DOT ${supported} CACHE BOOL
        "glslc supports GL_EXT_integer_dot_product (the int8 dot-product kernel is built)")
    message(STATUS "glslc supports GL_EXT_integer_dot_product: ${supported}")
endif()

function(vkml_add_shaders target)
    set(gen_root "${CMAKE_BINARY_DIR}/generated")
    set(gen_dir "${gen_root}/vkml_shaders")
    file(MAKE_DIRECTORY "${gen_dir}")

    set(headers "")
    foreach(shader IN LISTS ARGN)
        cmake_path(ABSOLUTE_PATH shader BASE_DIRECTORY "${PROJECT_SOURCE_DIR}" OUTPUT_VARIABLE src)
        cmake_path(GET src STEM name)
        set(spv "${gen_dir}/${name}.spv")
        set(dep "${gen_dir}/${name}.spv.d")
        set(hdr "${gen_dir}/${name}.spv.hpp")

        add_custom_command(
            OUTPUT "${spv}" "${hdr}"
            COMMAND "${VKML_GLSLC}"
                --target-env=vulkan1.2 -O -Werror
                -MD -MF "${dep}"
                -o "${spv}" "${src}"
            COMMAND "${CMAKE_COMMAND}"
                -DINPUT=${spv} -DOUTPUT=${hdr} -DSYMBOL=${name} -DSOURCE=${shader}
                -P "${VKML_EMBED_SPIRV_SCRIPT}"
            DEPENDS "${src}" "${VKML_EMBED_SPIRV_SCRIPT}"
            DEPFILE "${dep}"
            COMMENT "Compiling shader ${shader}"
            VERBATIM)
        list(APPEND headers "${hdr}")
    endforeach()

    add_custom_target(${target}_compile DEPENDS ${headers})
    add_library(${target} INTERFACE)
    add_dependencies(${target} ${target}_compile)
    target_include_directories(${target} INTERFACE "${gen_root}")
endfunction()
