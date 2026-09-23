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
// How b is laid out relative to a [batches, m, k].
struct Layout {
    std::size_t batches = 1;
    bool b_batched = false;     // b has its own [k, n] per batch rather than one shared
    bool b_transposed = false;  // b is stored as [n, k]
};

std::size_t count_mismatches(const std::vector<float>& got, const std::vector<float>& a,
                             const std::vector<float>& b, std::size_t m, std::size_t k,
                             std::size_t n, Layout layout = {}) {
    std::size_t bad = 0;
    for (std::size_t z = 0; z < layout.batches; ++z) {
        const float* az = a.data() + z * m * k;
        const float* bz = b.data() + (layout.b_batched ? z * k * n : 0);
        const float* gz = got.data() + z * m * n;
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                double exact = 0.0;
                double magnitude = 0.0;
                for (std::size_t p = 0; p < k; ++p) {
                    const float bv = layout.b_transposed ? bz[j * k + p] : bz[p * n + j];
                    const double term = double(az[i * k + p]) * double(bv);
                    exact += term;
                    magnitude += std::abs(term);
                }
                const double error = std::abs(double(gz[i * n + j]) - exact);
                const double bound = 2.0 * double(k) * 0x1p-24 * magnitude;
                if (error > bound + 1e-30) ++bad;
            }
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
    SECTION("b is a vector") {
        const Tensor b = Tensor::zeros(context, {3}, DType::F32);
        REQUIRE_THROWS_WITH(vkml::matmul(a, b), ContainsSubstring("at least 2"));
    }
    SECTION("dtype without a kernel") {
        // 16-bit weights are fine; activations must be f32 for now.
        const Tensor h = Tensor::zeros(context, {2, 3}, DType::F16);
        const Tensor b = Tensor::zeros(context, {3, 2}, DType::F32);
        REQUIRE_THROWS_WITH(vkml::matmul(h, b), ContainsSubstring("f16"));
        REQUIRE_THROWS_WITH(vkml::matmul(a, Tensor::zeros(context, {3, 2}, DType::I32)),
                            ContainsSubstring("i32"));
    }
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("matmul multiplies matching batches of matrices", "[matmul]") {
    vkml::Context context;
    const std::vector<float> a_host = random_values(2 * 3 * 5 * 7, 11);
    const std::vector<float> b_host = random_values(2 * 3 * 7 * 4, 12);
    const Tensor a = Tensor::from_data<float>(context, a_host, {2, 3, 5, 7});
    const Tensor b = Tensor::from_data<float>(context, b_host, {2, 3, 7, 4});

    const Tensor c = vkml::matmul(a, b);
    REQUIRE(c.shape() == vkml::Shape{2, 3, 5, 4});
    CHECK(count_mismatches(c.to_vector<float>(), a_host, b_host, 5, 7, 4,
                           {.batches = 6, .b_batched = true}) == 0);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("matmul_transposed multiplies by b stored as [n, k]", "[matmul]") {
    vkml::Context context;

    SECTION("one weight matrix, as in a linear layer") {
        const std::vector<float> a_host = random_values(3 * 33 * 70, 13);
        const std::vector<float> w_host = random_values(19 * 70, 14);
        const Tensor a = Tensor::from_data<float>(context, a_host, {3, 33, 70});
        const Tensor w = Tensor::from_data<float>(context, w_host, {19, 70});

        const Tensor c = vkml::matmul_transposed(a, w);
        REQUIRE(c.shape() == vkml::Shape{3, 33, 19});
        CHECK(count_mismatches(c.to_vector<float>(), a_host, w_host, 99, 70, 19,
                               {.b_transposed = true}) == 0);
    }
    SECTION("batched, as in attention scores q k^T") {
        const std::vector<float> q_host = random_values(4 * 9 * 16, 15);
        const std::vector<float> k_host = random_values(4 * 12 * 16, 16);
        const Tensor q = Tensor::from_data<float>(context, q_host, {4, 9, 16});
        const Tensor k = Tensor::from_data<float>(context, k_host, {4, 12, 16});

        const Tensor scores = vkml::matmul_transposed(q, k);
        REQUIRE(scores.shape() == vkml::Shape{4, 9, 12});
        CHECK(count_mismatches(scores.to_vector<float>(), q_host, k_host, 9, 16, 12,
                               {.batches = 4, .b_batched = true, .b_transposed = true}) == 0);
    }
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Batched matmul rejects batch shapes that differ", "[matmul]") {
    vkml::Context context;
    const Tensor a = Tensor::zeros(context, {2, 3, 4}, DType::F32);
    REQUIRE_THROWS_WITH(vkml::matmul(a, Tensor::zeros(context, {3, 4, 5}, DType::F32)),
                        ContainsSubstring("[2, 3, 4]") && ContainsSubstring("[3, 4, 5]"));
    REQUIRE_THROWS_WITH(vkml::matmul(a, Tensor::zeros(context, {1, 2, 4, 5}, DType::F32)),
                        ContainsSubstring("[1, 2, 4, 5]"));
    REQUIRE_THROWS_WITH(vkml::matmul_transposed(a, Tensor::zeros(context, {2, 5, 3}, DType::F32)),
                        ContainsSubstring("[2, 5, 3]"));
}

TEST_CASE("matmul_transposed with few rows, the matrix-vector shapes of decoding", "[matmul]") {
    vkml::Context context;
    // k multiples of 4 take vector loads; others the scalar path. n is not a
    // multiple of the outputs per workgroup.
    const auto [m, k, n] = GENERATE(table<std::size_t, std::size_t, std::size_t>({
        {1, 2048, 37},
        {1, 131, 70},
        {3, 64, 9},
        {4, 7, 1},
        {2, 1, 5},
    }));
    CAPTURE(m, k, n);

    const std::vector<float> a_host = random_values(m * k, 17);
    const std::vector<float> w_host = random_values(n * k, 18);
    const Tensor a = Tensor::from_data<float>(context, a_host, {std::int64_t(m), std::int64_t(k)});
    const Tensor w = Tensor::from_data<float>(context, w_host, {std::int64_t(n), std::int64_t(k)});

    const Tensor c = vkml::matmul_transposed(a, w);
    CHECK(count_mismatches(c.to_vector<float>(), a_host, w_host, m, k, n, {.b_transposed = true}) ==
          0);
    CHECK(context.validation_error_count() == 0);
}

namespace {

// Random multiples of 2^-7 in [-1, 1]: exact in f32, f16 and bf16 alike.
std::vector<std::uint16_t> random_halves(std::size_t n, std::uint32_t seed, DType dtype) {
    std::mt19937 rng{seed};
    std::uniform_int_distribution<int> dist{-128, 128};
    std::vector<std::uint16_t> bits(n);
    for (std::uint16_t& h : bits) {
        const float v = float(dist(rng)) / 128.0f;
        std::uint32_t f;
        std::memcpy(&f, &v, 4);
        if (dtype == DType::BF16) {
            h = std::uint16_t(f >> 16);  // exact: the low mantissa bits are zero
        } else if (v == 0.0f) {
            h = std::uint16_t((f >> 16) & 0x8000u);
        } else {  // a normal f16 with the same sign, exponent and top mantissa bits
            const std::uint32_t exponent = ((f >> 23) & 0xFFu) - 127 + 15;
            h = std::uint16_t(((f >> 16) & 0x8000u) | (exponent << 10) | ((f >> 13) & 0x3FFu));
        }
    }
    return bits;
}

Tensor half_tensor(vkml::Context& context, const std::vector<std::uint16_t>& bits,
                   vkml::Shape shape, DType dtype) {
    return Tensor::from_bytes(context, std::as_bytes(std::span{bits}), std::move(shape), dtype);
}

}  // namespace

TEST_CASE("matmul reads 16-bit weights exactly as their f32 widening", "[matmul]") {
    vkml::Context context;
    const DType dtype = GENERATE(DType::F16, DType::BF16);
    // Shapes for each path: vector and scalar matrix-vector loads, the tiled
    // kernel transposed and not, and grouped batches as in attention.
    const auto [m, k, n, transposed] =
        GENERATE(table<std::int64_t, std::int64_t, std::int64_t, bool>({
            {1, 64, 37, true},
            {2, 131, 9, true},
            {33, 70, 19, true},
            {5, 33, 7, false},
        }));
    CAPTURE(vkml::to_string(dtype), m, k, n, transposed);

    const Tensor a =
        Tensor::from_data<float>(context, random_values(std::size_t(m * k), 19), {m, k});
    const vkml::Shape w_shape = transposed ? vkml::Shape{n, k} : vkml::Shape{k, n};
    const Tensor w =
        half_tensor(context, random_halves(std::size_t(n * k), 20, dtype), w_shape, dtype);
    const Tensor w32 = vkml::cast(w, DType::F32);

    const auto product = [&](const Tensor& b) {
        return transposed ? vkml::matmul_transposed(a, b) : vkml::matmul(a, b);
    };
    CHECK(product(w).to_vector<float>() == product(w32).to_vector<float>());
    CHECK(context.validation_error_count() == 0);
}
