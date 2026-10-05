#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <vkml/vkml.hpp>

#include "io/host_quant.hpp"
#include "ops/internal.hpp"

using Catch::Matchers::ContainsSubstring;
using vkml::DType;
using vkml::Tensor;

namespace {

// The bits of a 16-bit tensor's elements.
std::vector<std::uint16_t> half_bits(const vkml::Tensor& t) {
    const std::vector<std::byte> bytes = t.to_bytes();
    std::vector<std::uint16_t> out(bytes.size() / 2);
    std::memcpy(out.data(), bytes.data(), out.size() * 2);
    return out;
}

std::vector<float> random_values(std::size_t n, std::uint32_t seed, float scale) {
    std::mt19937 rng{seed};
    std::normal_distribution<float> dist{0.0f, scale};
    std::vector<float> v(n);
    for (float& x : v) x = dist(rng);
    return v;
}

// A quantized matrix's scales, from f16, and for q4_1 its offsets too.
std::vector<float> scales_of(const vkml::QuantizedMatrix& q) {
    REQUIRE(q.scales.dtype() == DType::F16);
    std::vector<float> out;
    for (const std::uint16_t h : half_bits(q.scales)) {
        out.push_back(vkml::detail::f16_to_float(h));
    }
    return out;
}

}  // namespace

TEST_CASE("quantize_q8 stores each block of 32 as int8 over its absolute maximum", "[quantize]") {
    vkml::Context context;
    constexpr std::int64_t rows = 5, cols = 96;  // three blocks per row
    std::vector<float> w = random_values(rows * cols, 51, 0.1f);
    w[40] = 3.0f;  // an outlier: its block gets a coarser scale than the others
    const Tensor wt = Tensor::from_data<float>(context, w, {rows, cols});

    const vkml::QuantizedMatrix q = vkml::quantize_q8(wt);
    CHECK(q.rows == rows);
    CHECK(q.cols == cols);
    const std::vector<float> back = vkml::dequantize(q).to_vector<float>();
    REQUIRE(back.size() == w.size());

    // Scales are kept in f16, rounded from max |x| / 127, and the values
    // rounded to multiples of the rounded scale.
    const std::vector<float> scales = scales_of(q);
    for (std::int64_t r = 0; r < rows; ++r) {
        for (std::int64_t b = 0; b < cols / 32; ++b) {
            const auto begin = w.begin() + (r * cols + b * 32);
            float absmax = 0.0f;
            for (auto it = begin; it != begin + 32; ++it) absmax = std::max(absmax, std::abs(*it));
            const float scale = scales[std::size_t(r * cols / 32 + b)];
            CHECK(std::abs(scale - absmax / 127.0f) <= absmax / 127.0f * 0x1p-11f);
            for (std::int64_t i = 0; i < 32; ++i) {
                const std::size_t at = std::size_t(r * cols + b * 32 + i);
                CAPTURE(r, b, i, w[at], back[at], scale);
                const float code = back[at] / scale;
                CHECK(code == std::round(code));
                CHECK(std::abs(code) <= 127.0f);
                // Round to nearest: within half a step, plus float slack.
                CHECK(std::abs(back[at] - w[at]) <= 0.5f * scale * 1.0001f + 1e-12f);
            }
        }
    }
    // The block's absolute maximum itself comes back as 127 steps, within
    // f16's rounding of the step.
    CHECK(std::abs(back[40] - 3.0f) <= 3.0f * 0x1p-11f);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("quantize_q8 reads 16-bit weights and handles all-zero blocks", "[quantize]") {
    vkml::Context context;
    const std::vector<std::uint16_t> bf16(2 * 32, 0x3F80);  // 1.0
    std::vector<std::uint16_t> with_zero_block(3 * 32, 0x0000);
    std::fill(with_zero_block.begin() + 32, with_zero_block.end(), std::uint16_t{0xBF80});  // -1.0
    const Tensor a =
        Tensor::from_bytes(context, std::as_bytes(std::span{bf16}), {1, 64}, DType::BF16);
    const Tensor b = Tensor::from_bytes(context, std::as_bytes(std::span{with_zero_block}), {1, 96},
                                        DType::BF16);
    // 1 comes back as 127 steps of 1 / 127 rounded to f16.
    const float one =
        127.0f * vkml::detail::f16_to_float(vkml::detail::float_to_f16(1.0f / 127.0f));
    CHECK(one != 1.0f);
    CHECK(vkml::dequantize(vkml::quantize_q8(a)).to_vector<float>() == std::vector<float>(64, one));
    std::vector<float> want(96, -one);
    std::fill(want.begin(), want.begin() + 32, 0.0f);
    CHECK(vkml::dequantize(vkml::quantize_q8(b)).to_vector<float>() == want);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("quantize_q4 stores each block of 32 as 4-bit steps of its largest value / -8",
          "[quantize]") {
    vkml::Context context;
    constexpr std::int64_t rows = 3, cols = 64;
    std::vector<float> w = random_values(rows * cols, 54, 0.1f);
    w[70] = -2.0f;  // the largest magnitude of its block, negative
    w[101] = 1.5f;  // and of another, positive
    const Tensor wt = Tensor::from_data<float>(context, w, {rows, cols});

    const vkml::QuantizedMatrix q = vkml::quantize_q4(wt);
    CHECK(q.type == vkml::QuantType::q4_0);
    CHECK(q.rows == rows);
    CHECK(q.cols == cols);
    const std::vector<float> back = vkml::dequantize(q).to_vector<float>();
    REQUIRE(back.size() == w.size());

    for (std::int64_t b = 0; b < rows * cols / 32; ++b) {
        const auto begin = w.begin() + b * 32;
        const float largest = *std::max_element(
            begin, begin + 32, [](float x, float y) { return std::abs(x) < std::abs(y); });
        const float d = scales_of(q)[std::size_t(b)];
        CHECK(std::abs(d - largest / -8.0f) <= std::abs(largest / -8.0f) * 0x1p-11f);
        for (std::int64_t i = 0; i < 32; ++i) {
            const std::size_t at = std::size_t(b * 32 + i);
            CAPTURE(b, i, w[at], back[at], d);
            // The multiple of d from -8 to 7 nearest w. Values near -m clamp
            // to 7 steps, a whole step short, as in llama.cpp.
            const float want = std::clamp(std::round(w[at] / d), -8.0f, 7.0f) * d;
            CHECK(std::abs(back[at] - want) <= std::abs(d) * 1e-5f);
        }
    }
    // The largest value itself is -8 steps, exactly.
    CHECK(back[70] == -2.0f);
    CHECK(back[101] == 1.5f);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("quantize_q4 reads 16-bit weights and handles all-zero blocks", "[quantize]") {
    vkml::Context context;
    std::vector<std::uint16_t> bf16(2 * 32, 0x0000);
    std::fill(bf16.begin() + 32, bf16.end(), std::uint16_t{0xC000});  // -2.0
    const Tensor w =
        Tensor::from_bytes(context, std::as_bytes(std::span{bf16}), {1, 64}, DType::BF16);
    std::vector<float> want(64, -2.0f);
    std::fill(want.begin(), want.begin() + 32, 0.0f);
    CHECK(vkml::dequantize(vkml::quantize_q4(w)).to_vector<float>() == want);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("matmul_transposed with quantized weights multiplies by the dequantized weights",
          "[quantize]") {
    // With integer dot products, if the GPU has them, and without.
    const bool allow_dot = GENERATE(true, false);
    vkml::ContextOptions options;
    options.integer_dot_product = allow_dot;
    vkml::Context context{options};
    const vkml::QuantType type =
        GENERATE(vkml::QuantType::q8_0, vkml::QuantType::q4_0, vkml::QuantType::q4_1);
    const auto name = type == vkml::QuantType::q8_0   ? "q8_0"
                      : type == vkml::QuantType::q4_0 ? "q4_0"
                                                      : "q4_1";
    // Rows 1 and 5 take the matrix-vector kernels, 40 and 70 the tiled ones
    // (70 is not a multiple of their 64-row block).
    const std::int64_t m = GENERATE(1, 5, 40, 70);
    // gemv gives each output 32 lanes, each loading four words per iteration
    // and then one at a time: k = 1056 runs both loops for both formats.
    const std::int64_t k = 1056, n = 37;
    // With integer dot products, a is quantized to Q8_0 first and the
    // product is of that.
    const bool dot = context.device_info().integer_dot_product;
    CAPTURE(allow_dot, dot, name, m);
    const std::vector<float> a_host = random_values(std::size_t(m * k), 52, 1.0f);
    const Tensor a = Tensor::from_data<float>(context, a_host, {m, k});
    const std::vector<float> a_used =
        dot ? vkml::dequantize(vkml::detail::quantize_q8_f32(a)).to_vector<float>() : a_host;
    const Tensor w =
        Tensor::from_data<float>(context, random_values(std::size_t(n * k), 53, 0.2f), {n, k});

    const vkml::QuantizedMatrix q = type == vkml::QuantType::q8_0   ? vkml::quantize_q8(w)
                                    : type == vkml::QuantType::q4_0 ? vkml::quantize_q4(w)
                                                                    : vkml::quantize_q4_1(w);
    const std::vector<float> w_q = vkml::dequantize(q).to_vector<float>();
    const std::vector<float> got = vkml::matmul_transposed(a, q).to_vector<float>();

    // Against a double-precision product with the dequantized weights, within
    // the standard f32 dot-product rounding bound k * u * sum |a w| (doubled);
    // the kernels sum in their own order, so equality would be luck. (The
    // bound is far tighter than the effect of quantizing a, so this also
    // checks which path ran.) Q4_1 kernels sum scale * q and offset terms
    // apart, each at most 3 max |w| of the block (the offset is the block's
    // smallest value, the scale times 15 its range), so their bound uses that.
    std::vector<double> block_max(w_q.size() / 32, 0.0);
    for (std::size_t i = 0; i < w_q.size(); ++i) {
        block_max[i / 32] = std::max(block_max[i / 32], std::abs(double(w_q[i])));
    }
    std::size_t bad = 0;
    for (std::int64_t i = 0; i < m; ++i) {
        for (std::int64_t j = 0; j < n; ++j) {
            double exact = 0.0, magnitude = 0.0;
            for (std::int64_t p = 0; p < k; ++p) {
                const std::size_t at = std::size_t(j * k + p);
                const double term = double(a_used[std::size_t(i * k + p)]) * double(w_q[at]);
                exact += term;
                magnitude += type == vkml::QuantType::q4_1
                                 ? std::abs(double(a_used[std::size_t(i * k + p)])) * 3.0 *
                                       block_max[at / 32]
                                 : std::abs(term);
            }
            const double bound = 2.0 * double(k) * 0x1p-24 * magnitude;
            if (std::abs(double(got[std::size_t(i * n + j)]) - exact) > bound + 1e-30) ++bad;
        }
    }
    CHECK(bad == 0);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("quantize_q4_1 stores each block of 32 as 4-bit steps from its smallest value",
          "[quantize]") {
    vkml::Context context;
    constexpr std::int64_t rows = 3, cols = 64;
    std::vector<float> w = random_values(rows * cols, 55, 0.1f);
    w[5] = 2.0f;  // a skewed block: q4_0 would spend half its codes below the minimum
    const Tensor wt = Tensor::from_data<float>(context, w, {rows, cols});

    const vkml::QuantizedMatrix q = vkml::quantize_q4_1(wt);
    CHECK(q.type == vkml::QuantType::q4_1);
    CHECK(q.scales.shape() == vkml::Shape{rows, cols / 32, 2});  // (scale, offset) pairs
    const std::vector<float> back = vkml::dequantize(q).to_vector<float>();
    REQUIRE(back.size() == w.size());
    for (std::int64_t b = 0; b < rows * cols / 32; ++b) {
        const auto begin = w.begin() + b * 32;
        const float lo = *std::min_element(begin, begin + 32);
        const float hi = *std::max_element(begin, begin + 32);
        // (scale, offset) in f16: the offset is the smallest value rounded,
        // and values step from it by the rounded scale.
        const float d = scales_of(q)[std::size_t(2 * b)];
        const float offset = scales_of(q)[std::size_t(2 * b + 1)];
        CHECK(std::abs(offset - lo) <= std::abs(lo) * 0x1p-11f);
        CHECK(std::abs(d - (hi - lo) / 15.0f) <= (hi - lo) / 15.0f * 0x1p-11f);
        for (std::int64_t i = 0; i < 32; ++i) {
            const std::size_t at = std::size_t(b * 32 + i);
            CAPTURE(b, i, w[at], back[at], d);
            // Within half a step of the nearest of 16 evenly spaced values,
            // but past the ends, which rounding the offset and scale move.
            const float code = std::clamp(std::round((w[at] - offset) / d), 0.0f, 15.0f);
            CHECK(std::abs(back[at] - (code * d + offset)) <= d * 1e-5f);
        }
        CHECK(*std::min_element(back.begin() + b * 32, back.begin() + b * 32 + 32) == offset);
    }
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Quantized blocks of tiny values keep subnormal f16 scales", "[quantize]") {
    // A block whose largest value is 4e-4 has a q8_0 scale near 3.1e-6 and a
    // q4_0 one of 5e-5: below f16's smallest normal, 6.1e-5. Read as zero
    // (as some drivers convert f16 subnormals), the block would be.
    vkml::Context context;
    std::vector<float> w = random_values(64, 57, 1e-4f);
    for (std::size_t i = 0; i < 32; ++i) w[i] = std::clamp(w[i], -3.9e-4f, 3.9e-4f);
    w[3] = 4e-4f;
    w[40] = -3e-5f;  // the second block's largest: scales of 2.4e-7 and 3.8e-6
    for (std::size_t i = 32; i < 64; ++i) w[i] = std::clamp(w[i], -3e-5f, 3e-5f);
    const Tensor wt = Tensor::from_data<float>(context, w, {1, 64});
    for (const auto& q : {vkml::quantize_q8(wt), vkml::quantize_q4(wt)}) {
        CAPTURE(int(q.type));
        const std::vector<float> scales = scales_of(q);
        const std::vector<float> back = vkml::dequantize(q).to_vector<float>();
        for (std::size_t b = 0; b < 2; ++b) {
            CHECK(scales[b] != 0.0f);
            CHECK(std::abs(scales[b]) < 6.1e-5f);
            // Within half a step of the scale actually stored (and coarser
            // f16 subnormal steps still give 4 significant bits or more).
            const float limit = q.type == vkml::QuantType::q8_0 ? 0.5f : 1.0f;
            for (std::size_t i = 32 * b; i < 32 * b + 32; ++i) {
                CAPTURE(i, w[i], back[i]);
                CHECK(std::abs(back[i] - w[i]) <= limit * std::abs(scales[b]) * 1.0001f);
            }
        }
    }
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("quantize_q8 and quantize_q4 reject shapes they cannot block", "[quantize]") {
    vkml::Context context;
    REQUIRE_THROWS_WITH(vkml::quantize_q8(Tensor::zeros(context, {4, 40}, DType::F32)),
                        ContainsSubstring("multiple of 32"));
    REQUIRE_THROWS_WITH(vkml::quantize_q4(Tensor::zeros(context, {4, 40}, DType::F32)),
                        ContainsSubstring("multiple of 32"));
    REQUIRE_THROWS_WITH(vkml::quantize_q4(Tensor::zeros(context, {2, 4, 32}, DType::F32)),
                        ContainsSubstring("matrix"));
    REQUIRE_THROWS_WITH(vkml::quantize_q8(Tensor::zeros(context, {2, 4, 32}, DType::F32)),
                        ContainsSubstring("matrix"));
    const vkml::QuantizedMatrix q = vkml::quantize_q8(Tensor::zeros(context, {4, 64}, DType::F32));
    REQUIRE_THROWS_WITH(vkml::matmul_transposed(Tensor::zeros(context, {2, 32}, DType::F32), q),
                        ContainsSubstring("[2, 32]"));
}
