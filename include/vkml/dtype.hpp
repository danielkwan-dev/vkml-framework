#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace vkml {

// Element types a tensor can hold. F16 and BF16 have no portable C++ types
// yet, so their host-side data is passed as raw bytes. Kernels compute in f32
// and read 16-bit values where they accept them, such as matmul weights.
enum class DType : std::uint8_t { F32, F16, BF16, I32 };

constexpr std::size_t element_size(DType dtype) noexcept {
    switch (dtype) {
        case DType::F16:
        case DType::BF16: return 2;
        case DType::F32:
        case DType::I32: break;
    }
    return 4;
}

std::string_view to_string(DType dtype) noexcept;

}  // namespace vkml
