#include <cstdint>
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

// Row v of the table is v * 100 + column, so every value names its origin.
Tensor make_table(vkml::Context& context, std::int64_t vocab, std::int64_t dim) {
    std::vector<float> data(static_cast<std::size_t>(vocab * dim));
    for (std::int64_t v = 0; v < vocab; ++v) {
        for (std::int64_t d = 0; d < dim; ++d) data[std::size_t(v * dim + d)] = float(v * 100 + d);
    }
    return Tensor::from_data<float>(context, data, {vocab, dim});
}

}  // namespace

TEST_CASE("embedding gathers one table row per id, keeping the id shape", "[embedding]") {
    vkml::Context context;
    const Tensor table = make_table(context, 50, 3);
    const Tensor ids = Tensor::from_data<std::int32_t>(
        context, std::vector<std::int32_t>{7, 0, 49, 7, 1, 2}, {2, 3});

    const Tensor out = vkml::embedding(table, ids);
    CHECK(out.shape() == vkml::Shape{2, 3, 3});
    CHECK(out.to_vector<float>() == std::vector<float>{700, 701, 702, 0, 1, 2, 4900, 4901, 4902,
                                                       700, 701, 702, 100, 101, 102, 200, 201,
                                                       202});
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("embedding turns out-of-range ids into zero rows", "[embedding]") {
    vkml::Context context;
    const Tensor table = make_table(context, 4, 2);
    const Tensor ids =
        Tensor::from_data<std::int32_t>(context, std::vector<std::int32_t>{-1, 3, 4, 1000}, {4});
    CHECK(vkml::embedding(table, ids).to_vector<float>() ==
          std::vector<float>{0, 0, 300, 301, 0, 0, 0, 0});
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("embedding handles empty id tensors", "[embedding]") {
    vkml::Context context;
    const Tensor out =
        vkml::embedding(make_table(context, 4, 2), Tensor::zeros(context, {0}, DType::I32));
    CHECK(out.shape() == vkml::Shape{0, 2});
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("embedding rejects tables and ids of the wrong kind", "[embedding]") {
    vkml::Context context;
    const Tensor ids = Tensor::zeros(context, {2}, DType::I32);
    REQUIRE_THROWS_WITH(vkml::embedding(Tensor::zeros(context, {4}, DType::F32), ids),
                        ContainsSubstring("[4]"));
    REQUIRE_THROWS_WITH(
        vkml::embedding(make_table(context, 4, 2), Tensor::zeros(context, {2}, DType::F32)),
        ContainsSubstring("i32"));
    REQUIRE_THROWS_WITH(vkml::embedding(Tensor::zeros(context, {4, 2}, DType::I32), ids),
                        ContainsSubstring("i32 [4, 2]"));
}

TEST_CASE("embedding reads 16-bit tables exactly as their f32 widening", "[embedding]") {
    vkml::Context context;
    const DType dtype = GENERATE(DType::F16, DType::BF16);
    // 7 x 3 16-bit values: rows start mid-word, and the last word is half used.
    std::vector<std::uint16_t> bits(21);
    for (std::size_t i = 0; i < bits.size(); ++i) {
        bits[i] = std::uint16_t(dtype == DType::BF16 ? 0x3F80 + i : 0x3C00 + i);
    }
    const Tensor table = Tensor::from_bytes(context, std::as_bytes(std::span{bits}), {7, 3}, dtype);
    const Tensor ids =
        Tensor::from_data<std::int32_t>(context, std::vector<std::int32_t>{6, 1, 0, 3}, {4});

    CHECK(vkml::embedding(table, ids).to_vector<float>() ==
          vkml::embedding(vkml::cast(table, DType::F32), ids).to_vector<float>());
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("embedding reads quantized tables exactly as their dequantized values",
          "[embedding][quantize]") {
    vkml::Context context;
    const auto quantize = GENERATE(&vkml::quantize_q8, &vkml::quantize_q4, &vkml::quantize_q4_1);
    // 96 columns: three blocks per row, and rows of odd blocks for q4's words.
    std::vector<float> data(9 * 96);
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = float((i * 37) % 101) - 50.0f;
    const vkml::QuantizedMatrix table = quantize(Tensor::from_data<float>(context, data, {9, 96}));
    const Tensor ids = Tensor::from_data<std::int32_t>(
        context, std::vector<std::int32_t>{8, 0, 3, 3, 5, 1}, {2, 3});

    const Tensor out = vkml::embedding(table, ids);
    CHECK(out.shape() == vkml::Shape{2, 3, 96});
    CHECK(out.to_vector<float>() ==
          vkml::embedding(vkml::dequantize(table), ids).to_vector<float>());
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("embedding of a quantized table gives zero rows for out-of-range ids",
          "[embedding][quantize]") {
    vkml::Context context;
    const vkml::QuantizedMatrix table = vkml::quantize_q8(make_table(context, 4, 32));
    const Tensor ids =
        Tensor::from_data<std::int32_t>(context, std::vector<std::int32_t>{-1, 4, 2}, {3});
    const std::vector<float> out = vkml::embedding(table, ids).to_vector<float>();
    const std::vector<float> row2 = vkml::dequantize(table).to_vector<float>();
    CHECK(std::vector<float>(out.begin(), out.begin() + 64) == std::vector<float>(64, 0.0f));
    CHECK(std::vector<float>(out.begin() + 64, out.end()) ==
          std::vector<float>(row2.begin() + 64, row2.begin() + 96));
    CHECK(vkml::embedding(table, Tensor::zeros(context, {0}, DType::I32)).shape() ==
          vkml::Shape{0, 32});
    REQUIRE_THROWS_WITH(vkml::embedding(table, Tensor::zeros(context, {2}, DType::F32)),
                        ContainsSubstring("i32"));
    CHECK(context.validation_error_count() == 0);
}
