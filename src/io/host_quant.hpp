#pragma once

// Decoding and quantizing weights on the host, for the file readers: a
// matrix too large for one GPU buffer (Gemma 3 4B's q6_K embeddings, Gemma 2
// 2B's bf16 ones) cannot be uploaded whole and quantized there.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

#include "vkml/context.hpp"
#include "vkml/ops.hpp"

namespace vkml::detail {

// f16 bits as the f32 they stand for, exactly.
float f16_to_float(std::uint16_t h);

// Runs body(begin, end) over [0, n) in slices across the host's cores: a
// large file decodes hundreds of millions of values on loading, which take
// one core many seconds.
template <class Body>
void parallel_for(std::size_t n, const Body& body) {
    const std::size_t threads = std::clamp<std::size_t>(std::thread::hardware_concurrency(), 1, 16);
    const std::size_t per = std::max<std::size_t>((n + threads - 1) / threads, 1);
    std::vector<std::jthread> pool;
    for (std::size_t begin = per; begin < n; begin += per) {
        pool.emplace_back([&body, begin, end = std::min(n, begin + per)] { body(begin, end); });
    }
    body(0, std::min(n, per));
}

// A [rows, cols] matrix quantized to Q8_0 as shaders/quantize.comp does, on
// the host and every core: decode(u, y) writes the matrix's values u * unit
// .. u * unit + unit - 1, in row-major order, to y. unit divides 256 and is a
// multiple of 32, as does cols; decode is called from several threads.
QuantizedMatrix quantize_q8_host(Context& context, std::int64_t rows, std::int64_t cols,
                                 std::size_t unit,
                                 const std::function<void(std::size_t, float*)>& decode);

}  // namespace vkml::detail
