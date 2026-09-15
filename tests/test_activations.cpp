#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <vkml/vkml.hpp>

using Catch::Matchers::ContainsSubstring;
using vkml::DType;
using vkml::Tensor;

namespace {

// -20..20 in small steps, plus extremes where exp overflows and a naive
// tanh returns inf/inf = NaN.
std::vector<float> inputs() {
    std::vector<float> v;
    for (int i = -2000; i <= 2000; ++i) v.push_back(static_cast<float>(i) * 0.01f);
    for (float x : {-1e4f, -100.0f, -88.8f, 88.8f, 100.0f, 1e4f}) v.push_back(x);
    return v;
}

double silu(double x) { return x / (1.0 + std::exp(-x)); }

double gelu(double x) {
    const double c = std::sqrt(2.0 / 3.141592653589793);
    return 0.5 * x * (1.0 + std::tanh(c * (x + 0.044715 * x * x * x)));
}

// GPU exp and tanh are approximate (Vulkan allows several ulp), so compare
// with a relative tolerance plus a small absolute floor for results near 0.
std::size_t count_mismatches(const std::vector<float>& got, const std::vector<float>& x,
                             const std::function<double(double)>& f) {
    std::size_t bad = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double want = f(double(x[i]));
        if (!std::isfinite(got[i]) || std::abs(got[i] - want) > 1e-5 * std::abs(want) + 1e-6) {
            ++bad;
        }
    }
    return bad;
}

}  // namespace

TEST_CASE("silu matches x * sigmoid(x) and stays finite at the extremes", "[activations]") {
    vkml::Context context;
    const std::vector<float> x = inputs();
    const Tensor t = Tensor::from_data<float>(context, x, {std::int64_t(x.size())});

    const Tensor y = vkml::silu(t);
    CHECK(y.shape() == t.shape());
    CHECK(count_mismatches(y.to_vector<float>(), x, silu) == 0);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("gelu matches the tanh approximation and stays finite at the extremes",
          "[activations]") {
    vkml::Context context;
    const std::vector<float> x = inputs();
    const Tensor t = Tensor::from_data<float>(context, x, {std::int64_t(x.size())});

    CHECK(count_mismatches(vkml::gelu(t).to_vector<float>(), x, gelu) == 0);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Activations handle empty tensors and reject unsupported dtypes", "[activations]") {
    vkml::Context context;
    CHECK(vkml::silu(Tensor::zeros(context, {0}, DType::F32)).numel() == 0);
    REQUIRE_THROWS_WITH(vkml::gelu(Tensor::zeros(context, {2}, DType::I32)),
                        ContainsSubstring("i32"));
    CHECK(context.validation_error_count() == 0);
}
