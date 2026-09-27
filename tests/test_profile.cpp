#include <algorithm>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <vkml/vkml.hpp>

using vkml::DType;
using vkml::Tensor;

namespace {

const vkml::KernelTime* find(const std::vector<vkml::KernelTime>& profile, const std::string& name) {
    const auto it = std::ranges::find(profile, name, &vkml::KernelTime::name);
    return it == profile.end() ? nullptr : &*it;
}

}  // namespace

TEST_CASE("Profiling times each kernel on the GPU by name", "[profile]") {
    vkml::Context context;
    if (!context.device_info().timestamps) SKIP("the device has no compute timestamps");
    CHECK(context.profile().empty());

    context.set_profiling(true);
    const Tensor a = Tensor::zeros(context, {64, 256}, DType::F32);
    const Tensor b = Tensor::zeros(context, {256, 128}, DType::F32);
    Tensor c = vkml::matmul(a, b);
    for (int i = 0; i < 3; ++i) c = vkml::softmax(c);
    (void)c.to_bytes();  // waits, so the timestamps are in

    const auto profile = context.profile();
    const vkml::KernelTime* matmul = find(profile, "matmul");
    const vkml::KernelTime* softmax = find(profile, "softmax");
    REQUIRE(matmul != nullptr);
    REQUIRE(softmax != nullptr);
    CHECK(matmul->calls == 1);
    CHECK(softmax->calls == 3);
    CHECK(softmax->milliseconds > 0.0);
    CHECK(find(profile, "fill") != nullptr);  // zeros
    CHECK(find(profile, "copy") != nullptr);  // the readback
    // Slowest first.
    CHECK(std::ranges::is_sorted(profile, std::ranges::greater{}, &vkml::KernelTime::milliseconds));

    // Turning profiling off stops recording; reset_profile clears the totals.
    context.set_profiling(false);
    (void)vkml::softmax(c).to_bytes();
    CHECK(find(context.profile(), "softmax")->calls == 3);
    context.reset_profile();
    CHECK(context.profile().empty());
    CHECK(context.validation_error_count() == 0);
}
