#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <vkml/vkml.hpp>

using Catch::Matchers::ContainsSubstring;
using vkml::DType;
using vkml::Shape;
using vkml::Tensor;

namespace {

std::vector<float> iota_values(std::size_t n) {
    std::vector<float> v(n);
    std::iota(v.begin(), v.end(), 0.0f);
    return v;
}

// Host permute of a row-major array: out dimension i is input dimension dims[i].
std::vector<float> permute_reference(const std::vector<float>& x, const Shape& shape,
                                     const std::vector<int>& dims) {
    const std::size_t rank = shape.size();
    std::vector<std::size_t> in_strides(rank, 1);
    for (std::size_t i = rank - 1; i > 0; --i) {
        in_strides[i - 1] = in_strides[i] * std::size_t(shape[i]);
    }
    Shape out_shape(rank);
    for (std::size_t i = 0; i < rank; ++i) out_shape[i] = shape[std::size_t(dims[i])];

    std::vector<float> out(x.size());
    std::vector<std::size_t> index(rank, 0);
    for (std::size_t flat = 0; flat < out.size(); ++flat) {
        std::size_t src = 0;
        for (std::size_t i = 0; i < rank; ++i) src += index[i] * in_strides[std::size_t(dims[i])];
        out[flat] = x[src];
        for (std::size_t i = rank; i-- > 0;) {  // advance the output index, last dim fastest
            if (++index[i] < std::size_t(out_shape[i])) break;
            index[i] = 0;
        }
    }
    return out;
}

}  // namespace

TEST_CASE("transpose swaps two dimensions of a matrix", "[permute]") {
    vkml::Context context;
    const Tensor m =
        Tensor::from_data<float>(context, std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3});
    const Tensor t = vkml::transpose(m, 0, 1);
    CHECK(t.shape() == Shape{3, 2});
    CHECK(t.to_vector<float>() == std::vector<float>{1, 4, 2, 5, 3, 6});
    CHECK(vkml::transpose(m, -1, -2).to_vector<float>() == t.to_vector<float>());
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("permute rearranges dimensions of any rank", "[permute]") {
    vkml::Context context;

    SECTION("[seq, heads, head_dim] to [heads, seq, head_dim] for attention") {
        const Shape shape{7, 4, 16};
        const std::vector<float> x = iota_values(7 * 4 * 16);
        const Tensor t = Tensor::from_data<float>(context, x, shape);
        const Tensor p = vkml::permute(t, {1, 0, 2});
        CHECK(p.shape() == Shape{4, 7, 16});
        CHECK(p.to_vector<float>() == permute_reference(x, shape, {1, 0, 2}));
    }
    SECTION("a full reversal of rank 6, the largest supported") {
        const Shape shape{2, 3, 1, 4, 5, 3};
        const std::vector<float> x = iota_values(2 * 3 * 1 * 4 * 5 * 3);
        const Tensor t = Tensor::from_data<float>(context, x, shape);
        const Tensor p = vkml::permute(t, {5, 4, 3, 2, 1, 0});
        CHECK(p.shape() == Shape{3, 5, 4, 1, 3, 2});
        CHECK(p.to_vector<float>() == permute_reference(x, shape, {5, 4, 3, 2, 1, 0}));
    }
    SECTION("i32 values move unchanged") {
        const Tensor t = Tensor::from_data<std::int32_t>(
            context, std::vector<std::int32_t>{-1, 2, -3, 4}, {2, 2});
        CHECK(vkml::transpose(t, 0, 1).to_vector<std::int32_t>() ==
              std::vector<std::int32_t>{-1, -3, 2, 4});
    }
    SECTION("empty tensors") {
        CHECK(vkml::permute(Tensor::zeros(context, {3, 0}, DType::F32), {1, 0}).shape() ==
              Shape{0, 3});
    }
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("permute always returns fresh storage", "[permute]") {
    vkml::Context context;
    const Tensor t = Tensor::zeros(context, {2, 3}, DType::F32);
    CHECK_FALSE(vkml::permute(t, {0, 1}).shares_storage_with(t));
}

TEST_CASE("permute rejects dimension lists that are not a permutation", "[permute]") {
    vkml::Context context;
    const Tensor t = Tensor::zeros(context, {2, 3, 4}, DType::F32);
    REQUIRE_THROWS_WITH(vkml::permute(t, {0, 1}), ContainsSubstring("3 dimensions"));
    REQUIRE_THROWS_WITH(vkml::permute(t, {0, 1, 1}), ContainsSubstring("[0, 1, 1]"));
    REQUIRE_THROWS_WITH(vkml::permute(t, {0, 1, 3}), ContainsSubstring("[0, 1, 3]"));
    REQUIRE_THROWS_WITH(vkml::transpose(t, 0, 3), ContainsSubstring("3"));
    REQUIRE_THROWS_WITH(vkml::permute(Tensor::zeros(context, {1, 1, 1, 1, 1, 1, 1}, DType::F32),
                                      {0, 1, 2, 3, 4, 5, 6}),
                        ContainsSubstring("6"));
    REQUIRE_THROWS_WITH(vkml::transpose(Tensor::zeros(context, {2, 2}, DType::F16), 0, 1),
                        ContainsSubstring("f16"));
}
