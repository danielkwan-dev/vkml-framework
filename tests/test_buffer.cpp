#include <cstdint>
#include <cstring>
#include <utility>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <vkml/vkml.hpp>

#include "hal/buffer.hpp"

using Catch::Matchers::ContainsSubstring;
using vkml::hal::Buffer;
using vkml::hal::MemoryUsage;

TEST_CASE("Host-visible buffers are mapped and keep what the host writes", "[buffer]") {
    vkml::Context context;
    const auto usage = GENERATE(MemoryUsage::Upload, MemoryUsage::Readback);

    Buffer buffer{context.device(), 1024, usage};
    CHECK(buffer.handle() != VK_NULL_HANDLE);
    CHECK(buffer.size() == 1024);
    REQUIRE(buffer.mapped() != nullptr);

    const std::uint32_t pattern = 0xCAFEF00Du;
    std::memcpy(buffer.mapped() + 512, &pattern, sizeof pattern);
    buffer.flush();
    buffer.invalidate();
    std::uint32_t back = 0;
    std::memcpy(&back, buffer.mapped() + 512, sizeof back);
    CHECK(back == pattern);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Device-local buffers are not host-accessible", "[buffer]") {
    vkml::Context context;
    Buffer buffer{context.device(), 256, MemoryUsage::DeviceLocal};
    CHECK(buffer.handle() != VK_NULL_HANDLE);
    CHECK(buffer.mapped() == nullptr);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Buffer rejects sizes the device cannot bind", "[buffer]") {
    vkml::Context context;

    SECTION("zero bytes") {
        REQUIRE_THROWS_WITH(Buffer(context.device(), 0, MemoryUsage::DeviceLocal),
                            ContainsSubstring("size"));
    }
    SECTION("beyond maxStorageBufferRange") {
        const std::uint64_t too_big = context.device_info().max_storage_buffer_range + 1;
        REQUIRE_THROWS_WITH(Buffer(context.device(), too_big, MemoryUsage::DeviceLocal),
                            ContainsSubstring("maxStorageBufferRange"));
    }
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Moving a Buffer transfers the allocation", "[buffer]") {
    vkml::Context context;
    Buffer a{context.device(), 64, MemoryUsage::Upload};
    const VkBuffer handle = a.handle();

    Buffer b = std::move(a);
    CHECK(b.handle() == handle);
    CHECK(b.mapped() != nullptr);
    CHECK(a.handle() == VK_NULL_HANDLE);  // NOLINT(bugprone-use-after-move): moved-from state is specified

    Buffer c{context.device(), 64, MemoryUsage::Readback};
    c = std::move(b);  // releases c's own allocation
    CHECK(c.handle() == handle);
    CHECK(context.validation_error_count() == 0);
}
