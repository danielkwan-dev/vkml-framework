#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <vkml/vkml.hpp>

#include "ops/internal.hpp"

using Catch::Matchers::ContainsSubstring;
using vkml::DType;
using vkml::Shape;
using vkml::Tensor;

namespace {

std::vector<float> random_values(std::size_t n, std::uint32_t seed) {
    std::mt19937 rng{seed};
    std::normal_distribution<float> dist{0.0f, 1.0f};
    std::vector<float> v(n);
    for (float& x : v) x = dist(rng);
    return v;
}

// The first rows of each batch of a [batches, capacity, width] array.
std::vector<float> first_rows(const std::vector<float>& x, std::size_t batches,
                              std::size_t capacity, std::size_t width, std::size_t rows) {
    std::vector<float> out;
    for (std::size_t b = 0; b < batches; ++b) {
        const auto begin = x.begin() + std::ptrdiff_t(b * capacity * width);
        out.insert(out.end(), begin, begin + std::ptrdiff_t(rows * width));
    }
    return out;
}

}  // namespace

TEST_CASE("write_rows copies a block into rows of every batch", "[kv_cache]") {
    vkml::Context context;
    Tensor cache = Tensor::zeros(context, {2, 5, 3}, DType::F32);
    const Tensor block = Tensor::from_data<float>(
        context, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 2, 3});

    vkml::detail::write_rows(cache, block, 1);
    const std::vector<float> got = cache.to_vector<float>();
    const std::vector<float> want{0, 0, 0, 1, 2, 3, 4,  5,  6,  0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 7, 8, 9, 10, 11, 12, 0, 0, 0, 0, 0, 0};
    CHECK(got == want);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("read_rows copies rows of every batch out", "[kv_cache]") {
    vkml::Context context;
    std::vector<float> values(2 * 5 * 3);
    for (std::size_t i = 0; i < values.size(); ++i) values[i] = float(i);
    const Tensor src = Tensor::from_data<float>(context, values, {2, 5, 3});

    const Tensor rows = vkml::detail::read_rows(src, 1, 2);
    CHECK(rows.shape() == vkml::Shape{2, 2, 3});
    CHECK(rows.to_vector<float>() == std::vector<float>{3, 4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 23});
    REQUIRE_THROWS_WITH(vkml::detail::read_rows(src, 4, 2), ContainsSubstring("[2, 5, 3]"));
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("write_rows rejects blocks that do not fit the cache", "[kv_cache]") {
    vkml::Context context;
    const Tensor cache = Tensor::zeros(context, {2, 5, 3}, DType::F32);
    REQUIRE_THROWS_WITH(
        vkml::detail::write_rows(cache, Tensor::zeros(context, {2, 2, 3}, DType::F32), 4),
        ContainsSubstring("[2, 5, 3]"));
    REQUIRE_THROWS_WITH(
        vkml::detail::write_rows(cache, Tensor::zeros(context, {2, 2, 4}, DType::F32), 0),
        ContainsSubstring("[2, 2, 4]"));
    REQUIRE_THROWS_WITH(
        vkml::detail::write_rows(cache, Tensor::zeros(context, {3, 2, 3}, DType::F32), 0),
        ContainsSubstring("[3, 2, 3]"));
}

TEST_CASE("Attention over a partly filled cache equals attention over its filled rows",
          "[kv_cache]") {
    vkml::Context context;
    const bool causal = GENERATE(false, true);
    // Three queries, or one as when decoding; rows past kv_len hold values
    // that must not be read.
    const std::size_t tq = GENERATE(std::size_t{3}, std::size_t{1});
    CAPTURE(causal, tq);
    constexpr std::size_t heads = 6, kv_heads = 2, capacity = 20, kv_len = 11, d = 16;
    const auto i64 = [](std::size_t x) { return std::int64_t(x); };

    const std::vector<float> q = random_values(heads * tq * d, 41);
    const std::vector<float> k = random_values(kv_heads * capacity * d, 42);
    const std::vector<float> v = random_values(kv_heads * capacity * d, 43);
    const Tensor qt = Tensor::from_data<float>(context, q, {i64(heads), i64(tq), i64(d)});
    const Tensor k_cache =
        Tensor::from_data<float>(context, k, {i64(kv_heads), i64(capacity), i64(d)});
    const Tensor v_cache =
        Tensor::from_data<float>(context, v, {i64(kv_heads), i64(capacity), i64(d)});

    const Tensor from_cache = vkml::detail::attention(qt, k_cache, v_cache, i64(kv_len), causal);

    const Tensor k_prefix =
        Tensor::from_data<float>(context, first_rows(k, kv_heads, capacity, d, kv_len),
                                 {i64(kv_heads), i64(kv_len), i64(d)});
    const Tensor v_prefix =
        Tensor::from_data<float>(context, first_rows(v, kv_heads, capacity, d, kv_len),
                                 {i64(kv_heads), i64(kv_len), i64(d)});
    const Tensor direct = vkml::attention(qt, k_prefix, v_prefix, causal);

    // Same kernels on the same values, so the results agree exactly.
    CHECK(from_cache.to_vector<float>() == direct.to_vector<float>());
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Cached attention rejects kv_len beyond the cache", "[kv_cache]") {
    vkml::Context context;
    const Tensor q = Tensor::zeros(context, {2, 1, 8}, DType::F32);
    const Tensor cache = Tensor::zeros(context, {2, 4, 8}, DType::F32);
    REQUIRE_THROWS_WITH(vkml::detail::attention(q, cache, cache, 5, false), ContainsSubstring("5"));
}

TEST_CASE("Attention over many queries in chunks equals it in one go", "[kv_cache]") {
    vkml::Context context;
    const bool causal = GENERATE(false, true);
    CAPTURE(causal);
    // 70 queries after 30 cached keys, over 100 keys in all.
    constexpr std::size_t heads = 4, kv_heads = 2, tq = 70, kv_len = 100, d = 16;
    const auto i64 = [](std::size_t x) { return std::int64_t(x); };
    const Tensor q = Tensor::from_data<float>(context, random_values(heads * tq * d, 51),
                                              {i64(heads), i64(tq), i64(d)});
    const Tensor k = Tensor::from_data<float>(context, random_values(kv_heads * kv_len * d, 52),
                                              {i64(kv_heads), i64(kv_len), i64(d)});
    const Tensor v = Tensor::from_data<float>(context, random_values(kv_heads * kv_len * d, 53),
                                              {i64(kv_heads), i64(kv_len), i64(d)});

    const std::vector<float> whole =
        vkml::detail::attention(q, k, v, i64(kv_len), causal).to_vector<float>();
    // Room for the scores of 16 queries at a time: 5 chunks, the last partial.
    const std::int64_t budget = i64(heads * kv_len * 4 * 16);
    const std::vector<float> chunked =
        vkml::detail::attention(q, k, v, i64(kv_len), causal, budget).to_vector<float>();
    REQUIRE(chunked.size() == whole.size());
    std::size_t bad = 0;
    for (std::size_t i = 0; i < whole.size(); ++i) {
        if (!(std::abs(chunked[i] - whole[i]) <= 1e-6f * (1.0f + std::abs(whole[i])))) ++bad;
    }
    CHECK(bad == 0);
    CHECK(context.validation_error_count() == 0);
}
