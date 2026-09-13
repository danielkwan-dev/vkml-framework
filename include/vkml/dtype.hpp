#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace vkml {

// Element types a tensor can hold. F16 has no portable C++ type yet, so its
// host-side data is passed as raw bytes.
enum class DType : std::uint8_t { F32, F16, I32 };

constexpr std::size_t element_size(DType dtype) noexcept {
    switch (dtype) {
        case DType::F16: return 2;
        case DType::F32:
        case DType::I32: break;
    }
    return 4;
}

std::string_view to_string(DType dtype) noexcept;

}  // namespace vkml
