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

std::vector<float> ramp(std::size_t n, float start, float step) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = start + step * static_cast<float>(i);
    return v;
}

std::vector<float> reference(const std::vector<float>& a, const std::vector<float>& b,
                             const std::function<float(float, float)>& op) {
    std::vector<float> out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) out[i] = op(a[i], b[i]);
    return out;
}

}  // namespace

TEST_CASE("Elementwise ops match the host computation exactly", "[ops]") {
    vkml::Context context;
    // Not a multiple of the workgroup width; b never crosses zero, so div is finite.
    constexpr std::size_t n = 7 * 10'003;
    const std::vector<float> a_host = ramp(n, -3.0f, 0.001f);
    const std::vector<float> b_host = ramp(n, 0.5f, 0.0005f);
    const Tensor a = Tensor::from_data<float>(context, a_host, {7, 10'003});
    const Tensor b = Tensor::from_data<float>(context, b_host, {7, 10'003});

    // IEEE-754 requires correctly rounded + - * /, and Vulkan requires exact
    // results for the first three, so these compare bit-for-bit. Division is
    // allowed 2.5 ulp in Vulkan, hence the tolerance check below.
    CHECK(vkml::add(a, b).to_vector<float>() == reference(a_host, b_host, std::plus<>{}));
    CHECK(vkml::sub(a, b).to_vector<float>() == reference(a_host, b_host, std::minus<>{}));
    CHECK(vkml::mul(a, b).to_vector<float>() == reference(a_host, b_host, std::multiplies<>{}));

    const std::vector<float> quotient = vkml::div(a, b).to_vector<float>();
    const std::vector<float> expected = reference(a_host, b_host, std::divides<>{});
    std::size_t bad = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const float tolerance = 3e-7f * std::abs(expected[i]) + 1e-30f;
        if (std::abs(quotient[i] - expected[i]) > tolerance) ++bad;
    }
    CHECK(bad == 0);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Results keep the input shape in fresh storage", "[ops]") {
    vkml::Context context;
    const Tensor a = Tensor::zeros(context, {2, 3}, DType::F32);
    const Tensor c = vkml::add(a, a);
    CHECK(c.shape() == vkml::Shape{2, 3});
    CHECK(c.dtype() == DType::F32);
    CHECK_FALSE(c.shares_storage_with(a));
}

TEST_CASE("Ops chain on the GPU and temporaries may die before the result is read", "[ops]") {
    vkml::Context context;
    constexpr std::size_t n = 4096;
    const Tensor a = Tensor::from_data<float>(context, ramp(n, 1.0f, 1.0f), {n});
    const Tensor b = Tensor::from_data<float>(context, ramp(n, 2.0f, 0.0f), {n});

    // (a + b) * b - a: the intermediates are destroyed at the end of the
    // full-expression, before any of this work has been submitted.
    const Tensor out = vkml::sub(vkml::mul(vkml::add(a, b), b), a);

    const std::vector<float> got = out.to_vector<float>();
    for (std::size_t i = 0; i < n; ++i) {
        const float x = static_cast<float>(i + 1);
        REQUIRE(got[i] == (x + 2.0f) * 2.0f - x);
    }
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Inputs may be destroyed while the op using them is still pending", "[ops]") {
    vkml::Context context;
    std::vector<Tensor> results;
    for (int round = 0; round < 50; ++round) {
        const Tensor x = Tensor::from_data<float>(context, ramp(1024, float(round), 0.0f), {1024});
        results.push_back(vkml::add(x, x));
    }  // every x is gone here, but its add may not have run
    for (int round = 0; round < 50; ++round) {
        REQUIRE(results[static_cast<std::size_t>(round)].to_vector<float>() ==
                std::vector<float>(1024, 2.0f * float(round)));
    }
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Ops on empty tensors produce empty tensors", "[ops]") {
    vkml::Context context;
    const Tensor e = Tensor::zeros(context, {0, 5}, DType::F32);
    const Tensor r = vkml::mul(e, e);
    CHECK(r.shape() == vkml::Shape{0, 5});
    CHECK(r.to_vector<float>().empty());
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Elementwise ops reject mismatched operands", "[ops]") {
    vkml::Context context;
    const Tensor a = Tensor::zeros(context, {2, 3}, DType::F32);

    SECTION("different shapes") {
        const Tensor b = Tensor::zeros(context, {3, 2}, DType::F32);
        REQUIRE_THROWS_WITH(vkml::add(a, b),
                            ContainsSubstring("[2, 3]") && ContainsSubstring("[3, 2]"));
    }
    SECTION("different dtypes") {
        const Tensor b = Tensor::zeros(context, {2, 3}, DType::I32);
        REQUIRE_THROWS_WITH(vkml::mul(a, b), ContainsSubstring("i32"));
    }
    SECTION("a dtype without a kernel yet") {
        const Tensor h = Tensor::zeros(context, {2, 3}, DType::F16);
        REQUIRE_THROWS_WITH(vkml::add(h, h), ContainsSubstring("f16"));
    }
    CHECK(context.validation_error_count() == 0);
}
