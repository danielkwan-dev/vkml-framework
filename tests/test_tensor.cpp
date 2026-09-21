#include <cstdint>
#include <numeric>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <vkml/vkml.hpp>

#include "core/runtime.hpp"

using Catch::Matchers::ContainsSubstring;
using vkml::DType;
using vkml::Tensor;

TEST_CASE("DType reports its element size and name", "[tensor]") {
    CHECK(vkml::element_size(DType::F32) == 4);
    CHECK(vkml::element_size(DType::F16) == 2);
    CHECK(vkml::element_size(DType::I32) == 4);
    CHECK(vkml::to_string(DType::F16) == "f16");
}

TEST_CASE("zeros allocates the shape and clears every element", "[tensor]") {
    vkml::Context context;

    SECTION("f32, larger than one workgroup pass") {
        const Tensor t = Tensor::zeros(context, {37, 1000}, DType::F32);
        CHECK(t.shape() == vkml::Shape{37, 1000});
        CHECK(t.dtype() == DType::F32);
        CHECK(t.numel() == 37'000);
        CHECK(t.nbytes() == 148'000);
        CHECK(t.to_vector<float>() == std::vector<float>(37'000, 0.0f));
    }
    SECTION("f16 with an odd element count, so the last word is half used") {
        const Tensor t = Tensor::zeros(context, {5}, DType::F16);
        CHECK(t.nbytes() == 10);
        CHECK(t.to_bytes() == std::vector<std::byte>(10, std::byte{0}));
    }
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("zeros overwrites memory that previously held data", "[tensor]") {
    vkml::Context context;
    // Allocate, dirty and free a buffer so the next allocation likely reuses it.
    {
        const std::vector<float> ones(4096, 1.0f);
        const Tensor dirty = Tensor::from_data<float>(context, ones, {4096});
    }
    const Tensor t = Tensor::zeros(context, {4096}, DType::F32);
    CHECK(t.to_vector<float>() == std::vector<float>(4096, 0.0f));
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Host data round-trips through a tensor", "[tensor]") {
    vkml::Context context;

    SECTION("f32") {
        std::vector<float> data(3 * 4 * 5);
        std::iota(data.begin(), data.end(), -10.5f);
        const Tensor t = Tensor::from_data<float>(context, data, {3, 4, 5});
        CHECK(t.dtype() == DType::F32);
        CHECK(t.to_vector<float>() == data);
    }
    SECTION("i32") {
        const std::vector<std::int32_t> data{-2147483647 - 1, -1, 0, 1, 2147483647};
        const Tensor t = Tensor::from_data<std::int32_t>(context, data, {5});
        CHECK(t.dtype() == DType::I32);
        CHECK(t.to_vector<std::int32_t>() == data);
    }
    SECTION("raw f16 bits") {
        const std::vector<std::byte> bits{std::byte{0x00}, std::byte{0x3C},   // 1.0
                                          std::byte{0x00}, std::byte{0xC0},   // -2.0
                                          std::byte{0xFF}, std::byte{0x7B}};  // 65504
        const Tensor t = Tensor::from_bytes(context, bits, {3}, DType::F16);
        CHECK(t.to_bytes() == bits);
    }
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Scalars and empty tensors are valid shapes", "[tensor]") {
    vkml::Context context;

    const Tensor scalar = Tensor::from_data<float>(context, std::vector<float>{3.5f}, {});
    CHECK(scalar.numel() == 1);
    CHECK(scalar.to_vector<float>() == std::vector<float>{3.5f});

    const Tensor empty = Tensor::zeros(context, {4, 0, 2}, DType::F32);
    CHECK(empty.numel() == 0);
    CHECK(empty.nbytes() == 0);
    CHECK(empty.to_vector<float>().empty());
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Copying a Tensor shares its storage", "[tensor]") {
    vkml::Context context;
    const Tensor a = Tensor::zeros(context, {8}, DType::F32);
    const Tensor b = a;  // NOLINT(performance-unnecessary-copy-initialization): copy is the point
    CHECK(b.shares_storage_with(a));
    CHECK_FALSE(Tensor::zeros(context, {8}, DType::F32).shares_storage_with(a));
}

TEST_CASE("Tensor rejects inconsistent shapes, data and types", "[tensor]") {
    vkml::Context context;

    SECTION("negative dimension") {
        REQUIRE_THROWS_WITH(Tensor::zeros(context, {2, -1}, DType::F32),
                            ContainsSubstring("dimension"));
    }
    SECTION("data size does not match the shape") {
        const std::vector<float> data(5);
        REQUIRE_THROWS_WITH(Tensor::from_data<float>(context, data, {2, 3}),
                            ContainsSubstring("6 elements"));
    }
    SECTION("reading back as the wrong type") {
        const Tensor t = Tensor::zeros(context, {2}, DType::I32);
        REQUIRE_THROWS_WITH(t.to_vector<float>(), ContainsSubstring("i32"));
    }
    SECTION("larger than one GPU buffer can hold") {
        const std::int64_t too_many =
            static_cast<std::int64_t>(context.device_info().max_storage_buffer_range / 4 + 1);
        REQUIRE_THROWS_WITH(Tensor::empty(context, {too_many}, DType::F32),
                            ContainsSubstring("maxStorageBufferRange"));
    }
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("reshape reinterprets the same storage under a new shape", "[tensor]") {
    vkml::Context context;
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 0.0f);
    const Tensor t = Tensor::from_data<float>(context, data, {2, 3, 4});

    const Tensor r = t.reshape({6, 4});
    CHECK(r.shape() == vkml::Shape{6, 4});
    CHECK(r.shares_storage_with(t));
    CHECK(r.to_vector<float>() == data);

    CHECK(t.reshape({-1}).shape() == vkml::Shape{24});
    CHECK(t.reshape({4, -1, 3}).shape() == vkml::Shape{4, 2, 3});
    CHECK(Tensor::zeros(context, {0, 5}, DType::F32).reshape({5, -1, 2}).shape() ==
          vkml::Shape{5, 0, 2});
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("reshape rejects shapes with a different element count", "[tensor]") {
    vkml::Context context;
    const Tensor t = Tensor::zeros(context, {2, 3}, DType::F32);
    REQUIRE_THROWS_WITH(t.reshape({4, 2}),
                        ContainsSubstring("[2, 3]") && ContainsSubstring("[4, 2]"));
    REQUIRE_THROWS_WITH(t.reshape({4, -1}), ContainsSubstring("[4, -1]"));
    REQUIRE_THROWS_WITH(t.reshape({-1, -1}), ContainsSubstring("one -1"));
    REQUIRE_THROWS_WITH(Tensor::zeros(context, {0}, DType::F32).reshape({-1, 0}),
                        ContainsSubstring("ambiguous"));
}

TEST_CASE("Uploads free their staging memory before it piles up", "[tensor]") {
    vkml::Context context;
    vkml::detail::Runtime& runtime = context.runtime();
    runtime.set_retired_limit(1 << 20);  // 1 MiB instead of the default

    // 16 uploads of 256 KiB each record copies without ever waiting, so
    // without the limit 4 MiB of staging buffers would be held at once.
    const std::vector<float> data(64 * 1024, 1.0f);
    std::vector<Tensor> tensors;
    for (int i = 0; i < 16; ++i) {
        tensors.push_back(Tensor::from_data<float>(context, data, {64 * 1024}));
        CHECK(runtime.retired_bytes() <= (1u << 20));
    }
    CHECK(tensors.back().to_vector<float>() == data);
    CHECK(context.validation_error_count() == 0);
}
