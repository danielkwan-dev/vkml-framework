#include "core/runtime.hpp"

#include <algorithm>
#include <array>

#include <vkml_shaders/binary.spv.hpp>
#include <vkml_shaders/fill.spv.hpp>

namespace vkml::detail {

namespace {

struct FillParams {
    std::uint32_t count;
    std::uint32_t value;
};

struct BinaryParams {
    std::uint32_t count;
};

}  // namespace

Runtime::Runtime(const hal::DeviceConfig& config) : device(config), stream(device) {}

Runtime::~Runtime() {
    try {
        synchronize();
    } catch (...) {
        // Device lost: nothing is running any more, so freeing is still safe.
    }
}

void Runtime::retire(hal::Buffer buffer) {
    retired_.emplace_back(stream.pending_value(), std::move(buffer));
    collect();
}

void Runtime::synchronize() {
    stream.synchronize();
    retired_.clear();
}

void Runtime::collect() {
    const std::uint64_t done = stream.completed();
    std::erase_if(retired_, [done](const auto& entry) { return entry.first <= done; });
}

std::uint32_t Runtime::workgroup_width() const noexcept {
    const DeviceInfo& d = device.info();
    return std::min({256u, d.max_workgroup_size[0], d.max_workgroup_invocations});
}

std::uint32_t Runtime::workgroup_count(std::uint64_t count) const noexcept {
    const std::uint64_t width = workgroup_width();
    const std::uint64_t groups = std::max<std::uint64_t>(1, (count + width - 1) / width);
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(groups, device.info().max_workgroup_count[0]));
}

const hal::ComputePipeline& Runtime::fill() {
    if (!fill_) {
        const std::array<std::uint32_t, 1> spec{workgroup_width()};
        fill_ = std::make_unique<hal::ComputePipeline>(device, shaders::fill,
                                                       /*storage_buffer_count=*/1,
                                                       sizeof(FillParams), spec);
    }
    return *fill_;
}

const hal::ComputePipeline& Runtime::binary(BinaryOp op) {
    auto& pipeline = binary_[static_cast<std::size_t>(op)];
    if (!pipeline) {
        const std::array<std::uint32_t, 2> spec{workgroup_width(), static_cast<std::uint32_t>(op)};
        pipeline = std::make_unique<hal::ComputePipeline>(device, shaders::binary,
                                                          /*storage_buffer_count=*/3,
                                                          sizeof(BinaryParams), spec);
    }
    return *pipeline;
}

}  // namespace vkml::detail
