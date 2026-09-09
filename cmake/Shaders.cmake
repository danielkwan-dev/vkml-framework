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
