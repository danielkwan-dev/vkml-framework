#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <vkml/vkml.hpp>

using Catch::Matchers::ContainsSubstring;
using vkml::DType;
using vkml::Tensor;

namespace {

Tensor from_halves(vkml::Context& context, const std::vector<std::uint16_t>& bits, DType dtype) {
    return Tensor::from_bytes(context, std::as_bytes(std::span{bits}), {std::int64_t(bits.size())},
                              dtype);
}

float bits_to_float(std::uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

}  // namespace

TEST_CASE("cast widens f16 to f32 exactly, including subnormals and specials", "[cast]") {
    vkml::Context context;
    const float inf = std::numeric_limits<float>::infinity();
    // 1, -2, 65504 (max), 2^-24 (smallest subnormal), 1023 * 2^-24 (largest
    // subnormal), -0, inf, -inf; an odd count so the last word is half used.
    const std::vector<std::uint16_t> bits{0x3C00, 0xC000, 0x7BFF, 0x0001, 0x03FF,
                                          0x8000, 0x7C00, 0xFC00, 0x3555};
    const std::vector<float> got =
        vkml::cast(from_halves(context, bits, DType::F16), DType::F32).to_vector<float>();
    const std::vector<float> want{1.0f,  -2.0f, 65504.0f, 0x1p-24f,       1023 * 0x1p-24f,
                                  -0.0f, inf,   -inf,     0.333251953125f};
    CHECK(got == want);
    CHECK(std::signbit(got[5]));

    const std::vector<float> nan =
        vkml::cast(from_halves(context, {0x7E00}, DType::F16), DType::F32).to_vector<float>();
    CHECK(std::isnan(nan[0]));
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("cast widens bf16 to f32 exactly", "[cast]") {
    vkml::Context context;
    const std::vector<std::uint16_t> bits{0x3F80, 0xC049, 0x7E96, 0xFF80, 0x0001, 0x8000, 0x4049};
    const std::vector<float> got =
        vkml::cast(from_halves(context, bits, DType::BF16), DType::F32).to_vector<float>();
    for (std::size_t i = 0; i < bits.size(); ++i) {
        CHECK(got[i] == bits_to_float(std::uint32_t(bits[i]) << 16));
    }
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("cast keeps shape, copies f32, and rejects conversions it lacks", "[cast]") {
    vkml::Context context;
    const Tensor x = Tensor::from_data<float>(context, std::vector<float>{1, 2, 3, 4}, {2, 2});
    const Tensor y = vkml::cast(x, DType::F32);
    CHECK(y.shape() == vkml::Shape{2, 2});
    CHECK_FALSE(y.shares_storage_with(x));
    CHECK(y.to_vector<float>() == std::vector<float>{1, 2, 3, 4});
    CHECK(vkml::cast(Tensor::zeros(context, {0, 3}, DType::BF16), DType::F32).shape() ==
          vkml::Shape{0, 3});

    REQUIRE_THROWS_WITH(vkml::cast(x, DType::F16), ContainsSubstring("f32 to f16"));
    REQUIRE_THROWS_WITH(vkml::cast(Tensor::zeros(context, {2}, DType::I32), DType::F32),
                        ContainsSubstring("i32 to f32"));
    CHECK(vkml::to_string(DType::BF16) == "bf16");
    CHECK(vkml::element_size(DType::BF16) == 2);
}
