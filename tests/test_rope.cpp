#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <vkml/vkml.hpp>

using Catch::Matchers::ContainsSubstring;
using vkml::DType;
using vkml::RopeStyle;
using vkml::Shape;
using vkml::Tensor;

namespace {

std::vector<float> random_values(std::size_t n, std::uint32_t seed) {
    std::mt19937 rng{seed};
    std::uniform_real_distribution<float> dist{-1.0f, 1.0f};
    std::vector<float> v(n);
    for (float& x : v) x = dist(rng);
    return v;
}

// Rotates x [batch, seq, heads, dim] in double precision.
std::vector<double> rope_reference(const std::vector<float>& x, std::size_t batch, std::size_t seq,
                                   std::size_t heads, std::size_t dim, std::size_t start_pos,
                                   double theta, RopeStyle style) {
    std::vector<double> out(x.begin(), x.end());
    const std::size_t half = dim / 2;
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t s = 0; s < seq; ++s) {
            for (std::size_t h = 0; h < heads; ++h) {
                const std::size_t base = ((b * seq + s) * heads + h) * dim;
                for (std::size_t i = 0; i < half; ++i) {
                    const double angle =
                        double(start_pos + s) * std::pow(theta, -2.0 * double(i) / double(dim));
                    const std::size_t ia = style == RopeStyle::Interleaved ? 2 * i : i;
                    const std::size_t ib = style == RopeStyle::Interleaved ? 2 * i + 1 : i + half;
                    const double a = x[base + ia];
                    const double c = x[base + ib];
                    out[base + ia] = a * std::cos(angle) - c * std::sin(angle);
                    out[base + ib] = a * std::sin(angle) + c * std::cos(angle);
                }
            }
        }
    }
    return out;
}

std::size_t count_mismatches(const std::vector<float>& got, const std::vector<double>& want) {
    std::size_t bad = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (!(std::abs(got[i] - want[i]) <= 1e-6)) ++bad;  // inputs are in [-1, 1]
    }
    return bad;
}

}  // namespace

TEST_CASE("rope rotates pairs by position-dependent angles", "[rope]") {
    vkml::Context context;
    const RopeStyle style = GENERATE(RopeStyle::Interleaved, RopeStyle::RotateHalf);
    // start_pos far into a long context, where angles reach thousands of radians.
    const auto [batch, seq, heads, dim, start_pos] =
        GENERATE(table<std::size_t, std::size_t, std::size_t, std::size_t, std::size_t>({
            {1, 5, 4, 64, 0},
            {2, 3, 2, 128, 8000},
        }));
    CAPTURE(int(style), batch, seq, heads, dim, start_pos);
    const double theta = 10000.0;

    const std::vector<float> x = random_values(batch * seq * heads * dim, 21);
    const Shape shape{std::int64_t(batch), std::int64_t(seq), std::int64_t(heads),
                      std::int64_t(dim)};
    const Tensor xt = Tensor::from_data<float>(context, x, shape);
    const Tensor table = vkml::rope_table(context, 8192, std::int64_t(dim), float(theta));

    const Tensor out = vkml::rope(xt, table, std::int64_t(start_pos), style);
    CHECK(out.shape() == shape);
    CHECK(count_mismatches(out.to_vector<float>(), rope_reference(x, batch, seq, heads, dim,
                                                                  start_pos, theta, style)) == 0);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("rope at position 0 leaves the input unchanged", "[rope]") {
    vkml::Context context;
    const std::vector<float> x = random_values(1 * 1 * 3 * 8, 22);
    const Tensor xt = Tensor::from_data<float>(context, x, {1, 3, 8});
    const Tensor table = vkml::rope_table(context, 4, 8, 10000.0f);
    CHECK(vkml::rope(xt, table, 0, RopeStyle::RotateHalf).to_vector<float>() == x);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("rope_table stores cos and sin per position and frequency", "[rope]") {
    vkml::Context context;
    const Tensor table = vkml::rope_table(context, 3, 4, 100.0f);
    CHECK(table.shape() == Shape{3, 2, 2});
    const std::vector<float> t = table.to_vector<float>();
    // position 2, frequency 1: angle = 2 * 100^(-2/4) = 0.2
    CHECK(t[(2 * 2 + 1) * 2 + 0] == float(std::cos(0.2)));
    CHECK(t[(2 * 2 + 1) * 2 + 1] == float(std::sin(0.2)));
}

TEST_CASE("rope rejects mismatched shapes and positions beyond the table", "[rope]") {
    vkml::Context context;
    const Tensor table = vkml::rope_table(context, 16, 8, 10000.0f);
    const Tensor x = Tensor::zeros(context, {4, 2, 8}, DType::F32);

    REQUIRE_THROWS_WITH(vkml::rope(x, table, 13, RopeStyle::Interleaved), ContainsSubstring("16"));
    REQUIRE_THROWS_WITH(
        vkml::rope(Tensor::zeros(context, {4, 2, 6}, DType::F32), table, 0, RopeStyle::Interleaved),
        ContainsSubstring("[4, 2, 6]"));
    REQUIRE_THROWS_WITH(
        vkml::rope(Tensor::zeros(context, {2, 8}, DType::F32), table, 0, RopeStyle::Interleaved),
        ContainsSubstring("[..., seq, heads, head_dim]"));
    REQUIRE_THROWS_WITH(vkml::rope(x, table, -1, RopeStyle::Interleaved), ContainsSubstring("-1"));
    REQUIRE_THROWS_WITH(vkml::rope_table(context, 16, 7, 10000.0f), ContainsSubstring("even"));
}
