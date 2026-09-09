# Script mode: cmake -DINPUT=x.spv -DOUTPUT=x.spv.hpp -DSYMBOL=x -DSOURCE=... -P EmbedSpirv.cmake
#
# SPIR-V is a stream of 32-bit words; glslc writes them little-endian. Each
# group of four bytes becomes one word, so the array is correctly aligned for
# VkShaderModuleCreateInfo::pCode without any casting.

file(READ "${INPUT}" hex HEX)
string(LENGTH "${hex}" hex_len)
math(EXPR remainder "${hex_len} % 8")
if(hex_len EQUAL 0 OR NOT remainder EQUAL 0)
    message(FATAL_ERROR "${INPUT} is not a whole number of 32-bit words")
endif()

string(REGEX REPLACE "(..)(..)(..)(..)" "0x\\4\\3\\2\\1u, " words "${hex}")
string(REGEX REPLACE "((0x........u, ){8})" "\\1\n    " words "${words}")

file(WRITE "${OUTPUT}"
"// Generated from ${SOURCE} by cmake/EmbedSpirv.cmake. Do not edit.
#pragma once

#include <cstdint>

namespace vkml::shaders {

inline constexpr std::uint32_t ${SYMBOL}[] = {
    ${words}
};

}  // namespace vkml::shaders
")
