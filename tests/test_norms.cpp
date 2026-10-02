#include <algorithm>
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
using vkml::Tensor;

namespace {

std::vector<float> random_values(std::size_t n, std::uint32_t seed, float scale) {
    std::mt19937 rng{seed};
    std::normal_distribution<float> dist{0.0f, scale};
    std::vector<float> v(n);
    for (float& x : v) x = dist(rng);
    return v;
}

std::vector<double> softmax_reference(const std::vector<float>& x, std::size_t rows,
                                      std::size_t cols) {
    std::vector<double> out(x.size());
    for (std::size_t r = 0; r < rows; ++r) {
        double max = -std::numeric_limits<double>::infinity();
        for (std::size_t c = 0; c < cols; ++c) max = std::max(max, double(x[r * cols + c]));
        double sum = 0.0;
        for (std::size_t c = 0; c < cols; ++c) sum += std::exp(double(x[r * cols + c]) - max);
        for (std::size_t c = 0; c < cols; ++c) {
            out[r * cols + c] = std::exp(double(x[r * cols + c]) - max) / sum;
        }
    }
    return out;
}

// GPU exp is approximate, so relative error plus a tiny absolute floor.
std::size_t count_mismatches(const std::vector<float>& got, const std::vector<double>& want,
                             double rel, double abs) {
    std::size_t bad = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (!(std::abs(got[i] - want[i]) <= rel * std::abs(want[i]) + abs)) ++bad;
    }
    return bad;
}

}  // namespace

TEST_CASE("softmax matches a double-precision reference row by row", "[softmax]") {
    vkml::Context context;
    // One column, fewer columns than a workgroup, and more than one.
    const auto [rows, cols] = GENERATE(table<std::size_t, std::size_t>({
        {5, 1},
        {37, 100},
        {3, 3000},
    }));
    CAPTURE(rows, cols);

    const std::vector<float> x = random_values(rows * cols, 7, 3.0f);
    const Tensor t = Tensor::from_data<float>(context, x, {std::int64_t(rows), std::int64_t(cols)});
    const std::vector<float> got = vkml::softmax(t).to_vector<float>();

    CHECK(count_mismatches(got, softmax_reference(x, rows, cols), 1e-4, 1e-8) == 0);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("softmax is stable for large inputs and treats -inf as masked", "[softmax]") {
    vkml::Context context;
    const float inf = std::numeric_limits<float>::infinity();
    const std::vector<float> x{1000.0f, 1000.0f, -inf, 999.0f,  // exp(1000) would overflow
                               -inf,    0.0f,    -inf, -inf};
    const Tensor t = Tensor::from_data<float>(context, x, {2, 4});
    const std::vector<float> got = vkml::softmax(t).to_vector<float>();

    const double e = std::exp(-1.0);
    const std::vector<double> want{1 / (2 + e), 1 / (2 + e), 0, e / (2 + e), 0, 1, 0, 0};
    CHECK(count_mismatches(got, want, 1e-5, 1e-8) == 0);
    CHECK(got[2] == 0.0f);  // masked entries are exactly zero
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("softmax runs over the last dimension of any rank", "[softmax]") {
    vkml::Context context;
    const std::vector<float> x = random_values(2 * 3 * 50, 8, 1.0f);
    const Tensor t = Tensor::from_data<float>(context, x, {2, 3, 50});
    const Tensor y = vkml::softmax(t);
    CHECK(y.shape() == t.shape());
    CHECK(count_mismatches(y.to_vector<float>(), softmax_reference(x, 6, 50), 1e-4, 1e-8) == 0);

    CHECK(vkml::softmax(Tensor::zeros(context, {0, 4}, DType::F32)).numel() == 0);
    CHECK(vkml::softmax(Tensor::zeros(context, {4, 0}, DType::F32)).numel() == 0);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("argmax finds each row's largest entry, the first of equal ones", "[argmax]") {
    vkml::Context context;
    // One column, fewer columns than a workgroup, and Gemma's vocabulary.
    const auto [rows, cols] = GENERATE(table<std::size_t, std::size_t>({
        {5, 1},
        {37, 100},
        {2, 262144},
    }));
    CAPTURE(rows, cols);

    std::vector<float> x = random_values(rows * cols, 11, 3.0f);
    // Ties go to the first: the last row's maximum appears twice.
    if (cols > 1) x[x.size() - 1] = x[x.size() - 2] = 100.0f;
    const Tensor t = Tensor::from_data<float>(context, x, {std::int64_t(rows), std::int64_t(cols)});
    const Tensor got = vkml::argmax(t);
    CHECK(got.shape() == vkml::Shape{std::int64_t(rows)});
    CHECK(got.dtype() == DType::I32);

    std::vector<std::int32_t> want(rows);
    for (std::size_t r = 0; r < rows; ++r) {
        const auto row = x.begin() + std::ptrdiff_t(r * cols);
        want[r] = std::int32_t(std::max_element(row, row + std::ptrdiff_t(cols)) - row);
    }
    CHECK(got.to_vector<std::int32_t>() == want);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("argmax keeps leading dimensions and skips -inf", "[argmax]") {
    vkml::Context context;
    const float inf = std::numeric_limits<float>::infinity();
    const std::vector<float> x{-inf, -inf, -5.0f, -inf, 1.0f, inf, 2.0f, 3.0f};
    const Tensor got = vkml::argmax(Tensor::from_data<float>(context, x, {2, 1, 4}));
    CHECK(got.shape() == vkml::Shape{2, 1});
    CHECK(got.to_vector<std::int32_t>() == std::vector<std::int32_t>{2, 1});
    CHECK(vkml::argmax(Tensor::zeros(context, {0, 4}, DType::F32)).numel() == 0);
    CHECK_THROWS_WITH(vkml::argmax(Tensor::zeros(context, {4, 0}, DType::F32)),
                      ContainsSubstring("argmax"));
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("rms_norm matches a double-precision reference", "[rms_norm]") {
    vkml::Context context;
    const auto [rows, cols] = GENERATE(table<std::size_t, std::size_t>({
        {1, 4096},  // a LLaMA-sized hidden state
        {19, 77},
    }));
    CAPTURE(rows, cols);
    const float eps = 1e-5f;

    const std::vector<float> x = random_values(rows * cols, 9, 2.0f);
    const std::vector<float> w = random_values(cols, 10, 1.0f);
    const Tensor xt =
        Tensor::from_data<float>(context, x, {std::int64_t(rows), std::int64_t(cols)});
    const Tensor wt = Tensor::from_data<float>(context, w, {std::int64_t(cols)});
    const std::vector<float> got = vkml::rms_norm(xt, wt, eps).to_vector<float>();

    std::vector<double> want(x.size());
    for (std::size_t r = 0; r < rows; ++r) {
        double sum_sq = 0.0;
        for (std::size_t c = 0; c < cols; ++c) sum_sq += double(x[r * cols + c]) * x[r * cols + c];
        const double scale = 1.0 / std::sqrt(sum_sq / double(cols) + eps);
        for (std::size_t c = 0; c < cols; ++c) want[r * cols + c] = x[r * cols + c] * scale * w[c];
    }
    CHECK(count_mismatches(got, want, 1e-5, 1e-6) == 0);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("rms_norm of zeros is zero thanks to eps", "[rms_norm]") {
    vkml::Context context;
    const Tensor x = Tensor::zeros(context, {3, 8}, DType::F32);
    const Tensor w = Tensor::from_data<float>(context, std::vector<float>(8, 1.0f), {8});
    CHECK(vkml::rms_norm(x, w, 1e-6f).to_vector<float>() == std::vector<float>(24, 0.0f));
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("rms_norm rejects a weight that does not match the last dimension", "[rms_norm]") {
    vkml::Context context;
    const Tensor x = Tensor::zeros(context, {3, 8}, DType::F32);
    REQUIRE_THROWS_WITH(vkml::rms_norm(x, Tensor::zeros(context, {7}, DType::F32), 1e-6f),
                        ContainsSubstring("[7]") && ContainsSubstring("[3, 8]"));
    REQUIRE_THROWS_WITH(vkml::rms_norm(x, Tensor::zeros(context, {1, 8}, DType::F32), 1e-6f),
                        ContainsSubstring("[1, 8]"));
    REQUIRE_THROWS_WITH(vkml::softmax(Tensor::zeros(context, {}, DType::F32)),
                        ContainsSubstring("at least 1"));
}
