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
using vkml::Tensor;

namespace {

std::vector<float> random_values(std::size_t n, std::uint32_t seed) {
    std::mt19937 rng{seed};
    std::uniform_real_distribution<float> dist{-1.0f, 1.0f};
    std::vector<float> v(n);
    for (float& x : v) x = dist(rng);
    return v;
}

// Checks got against a double-precision product. The standard bound on an f32
// dot product's rounding error is k * u * sum |a||b| (u = 2^-24), independent
// of summation order; it scales with sum |a||b|, not the result, which can
// cancel to near zero. Doubled for slack, it still catches any wrong term.
std::size_t count_mismatches(const std::vector<float>& got, const std::vector<float>& a,
                             const std::vector<float>& b, std::size_t m, std::size_t k,
                             std::size_t n) {
    std::size_t bad = 0;
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double exact = 0.0;
            double magnitude = 0.0;
            for (std::size_t p = 0; p < k; ++p) {
                const double term = double(a[i * k + p]) * double(b[p * n + j]);
                exact += term;
                magnitude += std::abs(term);
            }
            const double error = std::abs(double(got[i * n + j]) - exact);
            const double bound = 2.0 * double(k) * 0x1p-24 * magnitude;
            if (error > bound + 1e-30) ++bad;
        }
    }
    return bad;
}

}  // namespace

TEST_CASE("matmul of small integer matrices is exact", "[matmul]") {
    vkml::Context context;
    const Tensor a =
        Tensor::from_data<float>(context, std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3});
    const Tensor b =
        Tensor::from_data<float>(context, std::vector<float>{7, 8, 9, 10, 11, 12}, {3, 2});

    const Tensor c = vkml::matmul(a, b);
    CHECK(c.shape() == vkml::Shape{2, 2});
    CHECK(c.to_vector<float>() == std::vector<float>{58, 64, 139, 154});
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("matmul matches a double-precision reference for awkward sizes", "[matmul]") {
    vkml::Context context;
    // Sizes that are not multiples of any tile width, plus the matrix-vector
    // shape (m = 1) that token-by-token decoding uses.
    const auto [m, k, n] = GENERATE(table<std::size_t, std::size_t, std::size_t>({
        {67, 131, 45},
        {1, 512, 300},
        {300, 1, 7},
        {128, 128, 128},
    }));
    CAPTURE(m, k, n);

    const std::vector<float> a_host = random_values(m * k, 1);
    const std::vector<float> b_host = random_values(k * n, 2);
    const Tensor a = Tensor::from_data<float>(context, a_host, {std::int64_t(m), std::int64_t(k)});
    const Tensor b = Tensor::from_data<float>(context, b_host, {std::int64_t(k), std::int64_t(n)});

    const Tensor c = vkml::matmul(a, b);
    REQUIRE(c.shape() == vkml::Shape{std::int64_t(m), std::int64_t(n)});
    CHECK(count_mismatches(c.to_vector<float>(), a_host, b_host, m, k, n) == 0);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("matmul treats leading dimensions of a as extra rows", "[matmul]") {
    vkml::Context context;
    const std::vector<float> a_host = random_values(2 * 3 * 40, 3);
    const std::vector<float> b_host = random_values(40 * 9, 4);
    const Tensor a = Tensor::from_data<float>(context, a_host, {2, 3, 40});
    const Tensor b = Tensor::from_data<float>(context, b_host, {40, 9});

    const Tensor c = vkml::matmul(a, b);
    REQUIRE(c.shape() == vkml::Shape{2, 3, 9});
    CHECK(count_mismatches(c.to_vector<float>(), a_host, b_host, 6, 40, 9) == 0);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("matmul with an empty inner dimension is all zeros", "[matmul]") {
    vkml::Context context;
    const Tensor a = Tensor::zeros(context, {3, 0}, DType::F32);
    const Tensor b = Tensor::zeros(context, {0, 4}, DType::F32);
    const Tensor c = vkml::matmul(a, b);
    CHECK(c.shape() == vkml::Shape{3, 4});
    CHECK(c.to_vector<float>() == std::vector<float>(12, 0.0f));

    const Tensor no_rows = vkml::matmul(Tensor::zeros(context, {0, 5}, DType::F32),
                                        Tensor::zeros(context, {5, 2}, DType::F32));
    CHECK(no_rows.shape() == vkml::Shape{0, 2});
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("matmul rejects shapes and dtypes it cannot multiply", "[matmul]") {
    vkml::Context context;
    const Tensor a = Tensor::zeros(context, {2, 3}, DType::F32);

    SECTION("inner dimensions differ") {
        const Tensor b = Tensor::zeros(context, {4, 2}, DType::F32);
        REQUIRE_THROWS_WITH(vkml::matmul(a, b),
                            ContainsSubstring("[2, 3]") && ContainsSubstring("[4, 2]"));
    }
    SECTION("a is not at least a matrix") {
        const Tensor v = Tensor::zeros(context, {3}, DType::F32);
        const Tensor b = Tensor::zeros(context, {3, 2}, DType::F32);
        REQUIRE_THROWS_WITH(vkml::matmul(v, b), ContainsSubstring("at least 2"));
    }
    SECTION("b is not a matrix") {
        const Tensor b = Tensor::zeros(context, {1, 3, 2}, DType::F32);
        REQUIRE_THROWS_WITH(vkml::matmul(a, b), ContainsSubstring("matrix"));
    }
    SECTION("dtype without a kernel") {
        const Tensor h = Tensor::zeros(context, {3, 2}, DType::F16);
        REQUIRE_THROWS_WITH(vkml::matmul(a, h), ContainsSubstring("f16"));
    }
    CHECK(context.validation_error_count() == 0);
}
