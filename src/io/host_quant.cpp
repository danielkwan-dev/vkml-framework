#include "io/host_quant.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <span>

#include "vkml/error.hpp"
#include "vkml/tensor.hpp"

namespace vkml::detail {

namespace {

// x rounded to the nearest integer, ties to even, as GLSL's roundEven, for
// |x| < 2^22: adding 1.5 * 2^23 leaves no fraction bits, so the addition
// itself rounds. A plain expression, where std::nearbyint is a library call
// per value (a third of Gemma 3 4B's load).
float round_even(float x) {
    constexpr float kMagic = 12582912.0f;
    return (x + kMagic) - kMagic;
}

}  // namespace

float f16_to_float(std::uint16_t h) {
    const std::uint32_t sign = std::uint32_t(h & 0x8000u) << 16;
    const std::uint32_t exponent = (h >> 10) & 0x1Fu, mantissa = h & 0x3FFu;
    std::uint32_t bits;
    if (exponent == 0) {  // zero or subnormal: mantissa * 2^-24, exact in f32
        float f = float(mantissa) * 5.9604644775390625e-8f;
        std::memcpy(&bits, &f, 4);
        bits |= sign;
    } else if (exponent == 31) {
        bits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

QuantizedMatrix quantize_q8_host(Context& context, std::int64_t rows, std::int64_t cols,
                                 std::size_t unit,
                                 const std::function<void(std::size_t, float*)>& decode) {
    const std::size_t count = std::size_t(rows * cols);
    if (unit == 0 || unit % 32 != 0 || 256 % unit != 0 || cols % 32 != 0 || count % unit != 0) {
        throw Error("quantize_q8_host: " + std::to_string(rows) + " x " + std::to_string(cols) +
                    " values do not split into whole units of " + std::to_string(unit) +
                    " and rows of whole blocks");
    }
    // Per 32 values, scale max |x| / 127 and codes round(x * 127 / max |x|),
    // four bytes to a word, low first.
    std::vector<std::uint32_t> words(count / 4);
    std::vector<float> scales(count / 32);
    parallel_for(count / unit, [&](std::size_t begin, std::size_t end) {
        std::array<float, 256> y;
        for (std::size_t u = begin; u < end; ++u) {
            decode(u, y.data());
            for (std::size_t g = 0; g < unit; g += 32) {
                const float* x = y.data() + g;
                float absmax = 0.0f;
                for (std::size_t i = 0; i < 32; ++i) absmax = std::max(absmax, std::abs(x[i]));
                const float inv = absmax > 0.0f ? 127.0f / absmax : 0.0f;
                const std::size_t group = (u * unit + g) / 32;
                scales[group] = absmax / 127.0f;
                for (std::size_t w = 0; w < 8; ++w) {
                    std::uint32_t word = 0;
                    for (std::size_t k = 0; k < 4; ++k) {
                        const float v = std::clamp(round_even(x[4 * w + k] * inv), -127.0f, 127.0f);
                        word |= (std::uint32_t(std::int32_t(v)) & 0xFFu) << (8 * k);
                    }
                    words[group * 8 + w] = word;
                }
            }
        }
    });
    return QuantizedMatrix{.values = Tensor::from_bytes(context, std::as_bytes(std::span{words}),
                                                        {rows, cols / 4}, DType::I32),
                           .scales = Tensor::from_data<float>(context, scales, {rows, cols / 32}),
                           .rows = rows,
                           .cols = cols,
                           .type = QuantType::q8_0};
}

}  // namespace vkml::detail
