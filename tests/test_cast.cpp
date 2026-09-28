#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
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

Tensor from_halves(vkml::Context& context, const std::vector<std::uint16_t>& bits, DType dtype) {
    return Tensor::from_bytes(context, std::as_bytes(std::span{bits}), {std::int64_t(bits.size())},
                              dtype);
}

float bits_to_float(std::uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

std::uint32_t float_bits(float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    return bits;
}

// f32 to f16 bits, rounding to nearest even: the reference narrowing.
std::uint16_t to_f16(float f) {
    const std::uint32_t bits = float_bits(f);
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t abs = bits & 0x7FFFFFFFu;
    if (abs > 0x7F800000u) return std::uint16_t(sign | 0x7E00u);   // NaN
    if (abs >= 0x477FF000u) return std::uint16_t(sign | 0x7C00u);  // rounds past 65504
    if (abs < 0x38800000u) {  // below 2^-14: subnormal or zero in f16
        const double scaled = double(bits_to_float(abs)) * 16777216.0;  // units of 2^-24
        return std::uint16_t(sign | std::uint32_t(std::nearbyint(scaled)));
    }
    const std::uint32_t rounded = abs + 0xFFFu + ((abs >> 13) & 1u);
    return std::uint16_t(sign | ((rounded - 0x38000000u) >> 13));
}

// f32 to bf16 bits, rounding to nearest even.
std::uint16_t to_bf16(float f) {
    const std::uint32_t bits = float_bits(f);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) return std::uint16_t((bits >> 16) | 0x40u);  // NaN
    return std::uint16_t((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

bool is_nan_half(std::uint16_t h, DType dtype) {
    return dtype == DType::F16 ? (h & 0x7C00u) == 0x7C00u && (h & 0x3FFu) != 0
                               : (h & 0x7F80u) == 0x7F80u && (h & 0x7Fu) != 0;
}

}  // namespace

TEST_CASE("cast narrows f32 to f16 and bf16, rounding to nearest even", "[cast]") {
    vkml::Context context;
    const DType dtype = GENERATE(DType::F16, DType::BF16);
    CAPTURE(vkml::to_string(dtype));
    std::vector<float> values{0.0f, -0.0f, 1.0f, -2.5f, 65504.0f, 65520.0f, 1e6f, -1e-9f,
                              std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::quiet_NaN(),
                              // halfway between two f16 values, to both even neighbours
                              bits_to_float(0x3F801000u), bits_to_float(0x3F803000u),
                              // f16 subnormals, and halfway below the smallest one
                              bits_to_float(0x33800000u), bits_to_float(0x33000000u),
                              bits_to_float(0x387FFFFFu), 1.0f / 3.0f};
    std::mt19937 rng{7};
    std::uniform_int_distribution<std::uint32_t> any;
    for (int i = 0; i < 20000; ++i) values.push_back(bits_to_float(any(rng)));
    // Around f16's range, where the rounding paths meet.
    std::uniform_real_distribution<float> exponent{-26.0f, 17.0f};
    for (int i = 0; i < 20000; ++i)
        values.push_back(std::ldexp(1.0f + float(i % 97) / 97.0f, int(exponent(rng))));
    values.push_back(3.0f);  // an odd count, so the last word holds one value

    const Tensor x = Tensor::from_data<float>(context, values, {std::int64_t(values.size())});
    const Tensor y = vkml::cast(x, dtype);
    CHECK(y.dtype() == dtype);
    std::vector<std::uint16_t> got(values.size());
    const auto bytes = y.to_bytes();
    REQUIRE(bytes.size() == got.size() * 2);
    std::memcpy(got.data(), bytes.data(), bytes.size());

    std::size_t bad = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        const std::uint16_t want = dtype == DType::F16 ? to_f16(values[i]) : to_bf16(values[i]);
        const bool ok = is_nan_half(want, dtype) ? is_nan_half(got[i], dtype) : got[i] == want;
        if (!ok && ++bad <= 5) {
            UNSCOPED_INFO("value " << values[i] << " (bits " << float_bits(values[i]) << "): got "
                                   << got[i] << ", want " << want);
        }
    }
    CHECK(bad == 0);
    CHECK(context.validation_error_count() == 0);
}

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

    REQUIRE_THROWS_WITH(vkml::cast(Tensor::zeros(context, {2}, DType::F16), DType::BF16),
                        ContainsSubstring("f16 to bf16"));
    REQUIRE_THROWS_WITH(vkml::cast(Tensor::zeros(context, {2}, DType::I32), DType::F32),
                        ContainsSubstring("i32 to f32"));
    CHECK(vkml::to_string(DType::BF16) == "bf16");
    CHECK(vkml::element_size(DType::BF16) == 2);
}
