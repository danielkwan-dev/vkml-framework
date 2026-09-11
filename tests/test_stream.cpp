#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <vkml/vkml.hpp>
#include <vkml_shaders/fill.spv.hpp>

#include "hal/buffer.hpp"
#include "hal/device.hpp"
#include "hal/pipeline.hpp"
#include "hal/stream.hpp"

using Catch::Matchers::ContainsSubstring;
using vkml::hal::Buffer;
using vkml::hal::ComputePipeline;
using vkml::hal::MemoryUsage;
using vkml::hal::Stream;

namespace {

struct FillParams {
    std::uint32_t count;
    std::uint32_t value;
};

std::uint32_t fill_width(const vkml::Context& context) {
    const vkml::DeviceInfo& d = context.device_info();
    return std::min({256u, d.max_workgroup_size[0], d.max_workgroup_invocations});
}

ComputePipeline make_fill(vkml::Context& context) {
    const std::array<std::uint32_t, 1> spec{fill_width(context)};
    return ComputePipeline{context.device(), vkml::shaders::fill, 1, sizeof(FillParams), spec};
}

void record_fill(Stream& stream, const ComputePipeline& fill, const Buffer& dst,
                 std::uint32_t count, std::uint32_t value, std::uint32_t groups) {
    const FillParams params{count, value};
    const std::array<const Buffer*, 1> buffers{&dst};
    stream.dispatch(fill, buffers, std::as_bytes(std::span{&params, 1}), {groups, 1, 1});
}

std::vector<std::uint32_t> read_words(Buffer& buffer, std::size_t count) {
    buffer.invalidate();
    std::vector<std::uint32_t> words(count);
    std::memcpy(words.data(), buffer.mapped(), count * sizeof(std::uint32_t));
    return words;
}

}  // namespace

TEST_CASE("The fill kernel writes every element and nothing past the end", "[stream]") {
    vkml::Context context;
    const ComputePipeline fill = make_fill(context);
    Stream stream{context.device()};

    // Not a multiple of the workgroup width, and a grid smaller than one group
    // per element, so both the bounds check and the grid-stride loop matter.
    constexpr std::uint32_t count = 10'007;
    Buffer out{context.device(), (count + 1) * sizeof(std::uint32_t), MemoryUsage::Readback};
    std::memset(out.mapped(), 0, out.size());
    out.flush();

    record_fill(stream, fill, out, count, 0x3F800000u /* 1.0f */, 3);
    stream.synchronize();

    const std::vector<std::uint32_t> words = read_words(out, count + 1);
    CHECK(std::all_of(words.begin(), words.end() - 1, [](std::uint32_t w) { return w == 0x3F800000u; }));
    CHECK(words.back() == 0);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Copies move data through device-local memory and back", "[stream]") {
    vkml::Context context;
    Stream stream{context.device()};
    constexpr std::size_t count = 4096;
    constexpr std::uint64_t bytes = count * sizeof(std::uint32_t);

    Buffer upload{context.device(), bytes, MemoryUsage::Upload};
    Buffer device{context.device(), bytes, MemoryUsage::DeviceLocal};
    Buffer readback{context.device(), bytes, MemoryUsage::Readback};

    std::vector<std::uint32_t> source(count);
    for (std::size_t i = 0; i < count; ++i) source[i] = static_cast<std::uint32_t>(i * 2654435761u);
    std::memcpy(upload.mapped(), source.data(), bytes);
    upload.flush();

    stream.copy(upload, device, bytes);
    stream.copy(device, readback, bytes);  // must observe the first copy
    stream.synchronize();

    CHECK(read_words(readback, count) == source);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("A kernel sees the result of the kernel recorded before it", "[stream]") {
    vkml::Context context;
    const ComputePipeline fill = make_fill(context);
    Stream stream{context.device()};
    constexpr std::uint32_t count = 1000;
    Buffer device{context.device(), count * sizeof(std::uint32_t), MemoryUsage::DeviceLocal};
    Buffer readback{context.device(), count * sizeof(std::uint32_t), MemoryUsage::Readback};

    record_fill(stream, fill, device, count, 7, 4);
    record_fill(stream, fill, device, count / 2, 9, 4);  // write-after-write
    stream.copy(device, readback, count * sizeof(std::uint32_t));
    stream.synchronize();

    const std::vector<std::uint32_t> words = read_words(readback, count);
    CHECK(std::all_of(words.begin(), words.begin() + count / 2, [](std::uint32_t w) { return w == 9; }));
    CHECK(std::all_of(words.begin() + count / 2, words.end(), [](std::uint32_t w) { return w == 7; }));
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("A stream is reusable after each submission", "[stream]") {
    vkml::Context context;
    const ComputePipeline fill = make_fill(context);
    Stream stream{context.device()};
    Buffer out{context.device(), 64 * sizeof(std::uint32_t), MemoryUsage::Readback};

    std::uint64_t last = 0;
    for (std::uint32_t round = 1; round <= 3; ++round) {
        record_fill(stream, fill, out, 64, round, 1);
        const std::uint64_t value = stream.submit();
        CHECK(value > last);
        last = value;
        stream.wait(value);
        CHECK(stream.completed() >= value);
        CHECK(read_words(out, 64) == std::vector<std::uint32_t>(64, round));
    }
    CHECK(stream.submit() == last);  // nothing recorded: nothing new to wait for
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Many dispatches in one submission outgrow a single descriptor pool", "[stream]") {
    vkml::Context context;
    const ComputePipeline fill = make_fill(context);
    Stream stream{context.device()};
    Buffer out{context.device(), sizeof(std::uint32_t), MemoryUsage::Readback};

    for (std::uint32_t i = 0; i < 2000; ++i) record_fill(stream, fill, out, 1, i, 1);
    stream.synchronize();

    CHECK(read_words(out, 1).front() == 1999);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Stream rejects dispatches that do not match the pipeline", "[stream]") {
    vkml::Context context;
    const ComputePipeline fill = make_fill(context);
    Stream stream{context.device()};
    Buffer a{context.device(), 16, MemoryUsage::DeviceLocal};
    const FillParams params{4, 0};
    const auto push = std::as_bytes(std::span{&params, 1});

    SECTION("wrong number of buffers") {
        const std::array<const Buffer*, 2> two{&a, &a};
        REQUIRE_THROWS_WITH(stream.dispatch(fill, two, push, {1, 1, 1}),
                            ContainsSubstring("storage buffers"));
    }
    SECTION("wrong push-constant size") {
        const std::array<const Buffer*, 1> one{&a};
        REQUIRE_THROWS_WITH(stream.dispatch(fill, one, push.first(4), {1, 1, 1}),
                            ContainsSubstring("push constants"));
    }
    SECTION("more workgroups than the device allows") {
        const std::array<const Buffer*, 1> one{&a};
        const std::uint32_t too_many = context.device_info().max_workgroup_count[0] + 1;
        REQUIRE_THROWS_WITH(stream.dispatch(fill, one, push, {too_many, 1, 1}),
                            ContainsSubstring("maxComputeWorkGroupCount"));
    }
    SECTION("a copy larger than either buffer") {
        Buffer small{context.device(), 8, MemoryUsage::DeviceLocal};
        REQUIRE_THROWS_WITH(stream.copy(a, small, 16), ContainsSubstring("copy"));
    }
    stream.synchronize();  // a rejected command leaves the stream usable
    CHECK(context.validation_error_count() == 0);
}
