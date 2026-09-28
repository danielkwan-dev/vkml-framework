#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <vkml/vkml.hpp>

using Catch::Matchers::ContainsSubstring;
using vkml::DType;
using vkml::Tensor;

namespace {

std::vector<float> random_values(std::size_t n, std::uint32_t seed, float scale) {
    std::mt19937 rng{seed};
    std::normal_distribution<float> dist{0.0f, scale};
    std::vector<float> v(n);
    for (float& x : v) x = dist(rng);
    return v;
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

    for (std::int64_t r = 0; r < rows; ++r) {
        for (std::int64_t b = 0; b < cols / 32; ++b) {
            const auto begin = w.begin() + (r * cols + b * 32);
            float absmax = 0.0f;
            for (auto it = begin; it != begin + 32; ++it) absmax = std::max(absmax, std::abs(*it));
            const float scale = absmax / 127.0f;
            for (std::int64_t i = 0; i < 32; ++i) {
                const std::size_t at = std::size_t(r * cols + b * 32 + i);
                CAPTURE(r, b, i, w[at], back[at], scale);
                // Round to nearest: within half a step, plus float slack.
                CHECK(std::abs(back[at] - w[at]) <= 0.5f * scale * 1.0001f + 1e-12f);
            }
        }
    }
    // The block's absolute maximum itself comes back exactly (as +-127 * scale).
    CHECK(std::abs(back[40] - 3.0f) <= 3.0f * 1e-6f);
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
    CHECK(vkml::dequantize(vkml::quantize_q8(a)).to_vector<float>() ==
          std::vector<float>(64, 1.0f));
    std::vector<float> want(96, -1.0f);
    std::fill(want.begin(), want.begin() + 32, 0.0f);
    CHECK(vkml::dequantize(vkml::quantize_q8(b)).to_vector<float>() == want);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("matmul_transposed with q8 weights equals it with the weights dequantized",
          "[quantize]") {
    vkml::Context context;
    // Rows 1 and 5 take the matrix-vector kernel, 40 the tiled one.
    const std::int64_t m = GENERATE(1, 5, 40);
    const std::int64_t k = 160, n = 37;
    CAPTURE(m);
    const Tensor a =
        Tensor::from_data<float>(context, random_values(std::size_t(m * k), 52, 1.0f), {m, k});
    const Tensor w =
        Tensor::from_data<float>(context, random_values(std::size_t(n * k), 53, 0.2f), {n, k});

    const vkml::QuantizedMatrix q = vkml::quantize_q8(w);
    const Tensor exact = vkml::matmul_transposed(a, vkml::dequantize(q));
    // Same arithmetic in the same order: the results agree bit for bit.
    CHECK(vkml::matmul_transposed(a, q).to_vector<float>() == exact.to_vector<float>());
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("quantize_q8 rejects shapes it cannot block", "[quantize]") {
    vkml::Context context;
    REQUIRE_THROWS_WITH(vkml::quantize_q8(Tensor::zeros(context, {4, 40}, DType::F32)),
                        ContainsSubstring("multiple of 32"));
    REQUIRE_THROWS_WITH(vkml::quantize_q8(Tensor::zeros(context, {2, 4, 32}, DType::F32)),
                        ContainsSubstring("matrix"));
    const vkml::QuantizedMatrix q = vkml::quantize_q8(Tensor::zeros(context, {4, 64}, DType::F32));
    REQUIRE_THROWS_WITH(vkml::matmul_transposed(Tensor::zeros(context, {2, 32}, DType::F32), q),
                        ContainsSubstring("[2, 32]"));
}
