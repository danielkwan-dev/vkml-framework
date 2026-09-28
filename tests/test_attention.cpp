#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <vkml/vkml.hpp>

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

// softmax(q k^T / sqrt(d) + mask) v per head, in double precision. Query i sits
// at position i + (tk - tq), so with a KV cache the new queries see every
// cached key; under a causal mask it sees keys at positions <= its own.
std::vector<double> attention_reference(const std::vector<float>& q, const std::vector<float>& k,
                                        const std::vector<float>& v, std::size_t heads,
                                        std::size_t kv_heads, std::size_t tq, std::size_t tk,
                                        std::size_t d, bool causal) {
    std::vector<double> out(heads * tq * d, 0.0);
    const std::size_t group = heads / kv_heads;
    const double scale = 1.0 / std::sqrt(double(d));
    for (std::size_t h = 0; h < heads; ++h) {
        const std::size_t kh = h / group;
        for (std::size_t i = 0; i < tq; ++i) {
            const std::size_t visible = causal ? i + (tk - tq) + 1 : tk;
            std::vector<double> scores(visible);
            double max = -std::numeric_limits<double>::infinity();
            for (std::size_t j = 0; j < visible; ++j) {
                double s = 0.0;
                for (std::size_t c = 0; c < d; ++c) {
                    s += double(q[(h * tq + i) * d + c]) * double(k[(kh * tk + j) * d + c]);
                }
                scores[j] = s * scale;
                max = std::max(max, scores[j]);
            }
            double sum = 0.0;
            for (double& s : scores) sum += (s = std::exp(s - max));
            for (std::size_t j = 0; j < visible; ++j) {
                for (std::size_t c = 0; c < d; ++c) {
                    out[(h * tq + i) * d + c] += scores[j] / sum * double(v[(kh * tk + j) * d + c]);
                }
            }
        }
    }
    return out;
}

}  // namespace

TEST_CASE("attention matches a double-precision reference", "[attention]") {
    vkml::Context context;
    const bool causal = GENERATE(false, true);
    const auto [heads, kv_heads, tq, tk, d] =
        GENERATE(table<std::size_t, std::size_t, std::size_t, std::size_t, std::size_t>({
            {4, 4, 13, 13, 32},  // prefill of a whole prompt
            {4, 4, 1, 40, 64},   // one decode step against a KV cache
            {8, 2, 5, 9, 16},    // grouped-query attention, 4 query heads per KV head
            // Decode steps take a fused kernel that splits the keys into
            // chunks: several chunks and a partial one, a single key, a wide
            // head, and Qwen2.5's 7 query heads per KV head.
            {8, 2, 1, 300, 64},
            {4, 4, 1, 1, 32},
            {2, 1, 1, 70, 128},
            {14, 2, 1, 257, 64},
            // Longer prompts take a fused kernel over blocks of 32 queries and
            // of keys: several of each and partial ones, queries after a cache
            // (tq < tk), and a wide head.
            {4, 2, 70, 70, 64},
            {6, 3, 45, 100, 32},
            {2, 1, 40, 75, 128},
        }));
    CAPTURE(causal, heads, kv_heads, tq, tk, d);

    const std::vector<float> q = random_values(heads * tq * d, 31);
    const std::vector<float> k = random_values(kv_heads * tk * d, 32);
    const std::vector<float> v = random_values(kv_heads * tk * d, 33);
    const auto i64 = [](std::size_t x) { return std::int64_t(x); };
    const Tensor qt = Tensor::from_data<float>(context, q, {i64(heads), i64(tq), i64(d)});
    const Tensor kt = Tensor::from_data<float>(context, k, {i64(kv_heads), i64(tk), i64(d)});
    const Tensor vt = Tensor::from_data<float>(context, v, {i64(kv_heads), i64(tk), i64(d)});

    const Tensor out = vkml::attention(qt, kt, vt, causal);
    REQUIRE(out.shape() == Shape{i64(heads), i64(tq), i64(d)});

    const std::vector<float> got = out.to_vector<float>();
    const std::vector<double> want =
        attention_reference(q, k, v, heads, kv_heads, tq, tk, d, causal);
    std::size_t bad = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (!(std::abs(got[i] - want[i]) <= 1e-5 * (1.0 + std::abs(want[i])))) ++bad;
    }
    CHECK(bad == 0);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("causal attention lets the first query see only the first key", "[attention]") {
    vkml::Context context;
    // With one head of width 1, the output of query i is a weighted mean of
    // v[0..i], so query 0 must return v[0] exactly whatever the scores.
    const Tensor q = Tensor::from_data<float>(context, std::vector<float>{1, 2, 3}, {1, 3, 1});
    const Tensor k = Tensor::from_data<float>(context, std::vector<float>{3, 2, 1}, {1, 3, 1});
    const Tensor v = Tensor::from_data<float>(context, std::vector<float>{10, 20, 30}, {1, 3, 1});
    CHECK(vkml::attention(q, k, v, true).to_vector<float>()[0] == 10.0f);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("attention rejects inconsistent q, k and v", "[attention]") {
    vkml::Context context;
    const auto zeros = [&](Shape s) { return Tensor::zeros(context, std::move(s), DType::F32); };

    REQUIRE_THROWS_WITH(
        vkml::attention(zeros({4, 2, 8}), zeros({3, 5, 8}), zeros({3, 5, 8}), false),
        ContainsSubstring("multiple"));
    REQUIRE_THROWS_WITH(
        vkml::attention(zeros({4, 2, 8}), zeros({4, 5, 8}), zeros({4, 6, 8}), false),
        ContainsSubstring("[4, 6, 8]"));
    REQUIRE_THROWS_WITH(
        vkml::attention(zeros({4, 2, 8}), zeros({4, 5, 4}), zeros({4, 5, 4}), false),
        ContainsSubstring("[4, 5, 4]"));
    REQUIRE_THROWS_WITH(vkml::attention(zeros({2, 8}), zeros({2, 8}), zeros({2, 8}), false),
                        ContainsSubstring("[heads, seq, head_dim]"));
    REQUIRE_THROWS_WITH(vkml::attention(zeros({4, 6, 8}), zeros({4, 5, 8}), zeros({4, 5, 8}), true),
                        ContainsSubstring("more queries"));
}
