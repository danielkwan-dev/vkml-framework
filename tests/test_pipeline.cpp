#include <algorithm>
#include <array>
#include <span>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <vkml/vkml.hpp>
#include <vkml_shaders/fill.spv.hpp>

#include "hal/device.hpp"
#include "hal/pipeline.hpp"

using Catch::Matchers::ContainsSubstring;

namespace {

struct FillParams {
    std::uint32_t count;
    std::uint32_t value;
};

}  // namespace

TEST_CASE("Build-time embedding produces well-formed SPIR-V", "[pipeline]") {
    const std::span<const std::uint32_t> code{vkml::shaders::fill};
    REQUIRE(code.size() > 5);  // 5-word header plus at least one instruction
    CHECK(code[0] == vkml::hal::kSpirvMagic);
}

TEST_CASE("The fill kernel compiles into a compute pipeline on the device", "[pipeline]") {
    vkml::Context context;
    const vkml::DeviceInfo& d = context.device_info();

    // Workgroup width from the device, as every kernel will do it.
    const std::uint32_t width =
        std::min({256u, d.max_workgroup_size[0], d.max_workgroup_invocations});
    const std::array<std::uint32_t, 1> spec{width};

    vkml::hal::ComputePipeline pipeline{context.device(), vkml::shaders::fill,
                                        /*storage_buffer_count=*/1, sizeof(FillParams), spec};
    CHECK(pipeline.pipeline() != VK_NULL_HANDLE);
    CHECK(pipeline.layout() != VK_NULL_HANDLE);
    CHECK(context.validation_error_count() == 0);
}

// Guards the guard: every other test asserts validation_error_count() == 0,
// which proves nothing unless the counter demonstrably sees real errors.
TEST_CASE("Validation errors are caught and counted", "[pipeline][validation]") {
    vkml::Context context;
    if (!context.device_info().validation_enabled) SKIP("validation layers not installed");

    // A zero-size buffer (VUID-VkBufferCreateInfo-size-00912) is flagged by
    // every layer version. Subtler misuse is not: newer layers stopped
    // reporting a pipeline layout that lacks the shader's push-constant range.
    const VkBufferCreateInfo info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                  .pNext = nullptr,
                                  .flags = 0,
                                  .size = 0,
                                  .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                  .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                                  .queueFamilyIndexCount = 0,
                                  .pQueueFamilyIndices = nullptr};
    const VkDevice device = context.device().device();
    VkBuffer buffer = VK_NULL_HANDLE;
    if (vkCreateBuffer(device, &info, nullptr, &buffer) == VK_SUCCESS) {
        vkDestroyBuffer(device, buffer, nullptr);
    }
    CHECK(context.validation_error_count() > 0);
}

TEST_CASE("ComputePipeline rejects bad inputs before touching the driver", "[pipeline]") {
    vkml::Context context;

    SECTION("code that is not SPIR-V") {
        const std::array<std::uint32_t, 8> garbage{0xDEADBEEF};
        REQUIRE_THROWS_WITH(vkml::hal::ComputePipeline(context.device(), garbage, 1, 8),
                            ContainsSubstring("not SPIR-V"));
    }
    SECTION("push constants beyond the device limit") {
        const std::uint32_t too_big = context.device_info().max_push_constant_bytes + 4;
        REQUIRE_THROWS_WITH(
            vkml::hal::ComputePipeline(context.device(), vkml::shaders::fill, 1, too_big),
            ContainsSubstring("push constants"));
    }
    SECTION("push constants that are not a multiple of 4") {
        REQUIRE_THROWS_WITH(vkml::hal::ComputePipeline(context.device(), vkml::shaders::fill, 1, 6),
                            ContainsSubstring("multiple of 4"));
    }
    CHECK(context.validation_error_count() == 0);
}
