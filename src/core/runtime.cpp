#include "core/runtime.hpp"

#include <algorithm>
#include <array>

#include <vkml_shaders/fill.spv.hpp>

namespace vkml::detail {

namespace {

struct FillParams {
    std::uint32_t count;
    std::uint32_t value;
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

void Runtime::fill_zeros(const hal::Buffer& buffer) {
    const auto words = static_cast<std::uint32_t>(buffer.size() / 4);
    const FillParams params{words, 0u};
    const std::array<const hal::Buffer*, 1> buffers{&buffer};
    stream.dispatch(pipeline(shaders::fill, 1, sizeof(FillParams), {workgroup_width()}), buffers,
                    std::as_bytes(std::span{&params, 1}), {workgroup_count(words), 1, 1});
}

std::uint32_t Runtime::workgroup_width() const noexcept {
    const DeviceInfo& d = device.info();
    return std::min({256u, d.max_workgroup_size[0], d.max_workgroup_invocations});
}

std::uint32_t Runtime::workgroup_count(std::uint64_t count) const noexcept {
    const std::uint64_t width = workgroup_width();
    const std::uint64_t groups = std::max<std::uint64_t>(1, (count + width - 1) / width);
    return static_cast<std::uint32_t>(
        std::min<std::uint64_t>(groups, device.info().max_workgroup_count[0]));
}

const hal::ComputePipeline& Runtime::pipeline(std::span<const std::uint32_t> spirv,
                                              std::uint32_t storage_buffer_count,
                                              std::uint32_t push_constant_bytes,
                                              std::initializer_list<std::uint32_t> specialization) {
    // Kernels are static arrays, so the address identifies the kernel.
    auto& slot = pipelines_[PipelineKey{spirv.data(), specialization}];
    if (!slot) {
        const std::vector<std::uint32_t> spec{specialization};
        slot = std::make_unique<hal::ComputePipeline>(device, spirv, storage_buffer_count,
                                                      push_constant_bytes, spec);
    }
    return *slot;
}

std::uint32_t Runtime::matmul_tile() const noexcept {
    const DeviceInfo& d = device.info();
    const bool fits16 = d.max_workgroup_invocations >= 256 && d.max_workgroup_size[0] >= 16 &&
                        d.max_workgroup_size[1] >= 16;
    return fits16 ? 16 : 8;
}

}  // namespace vkml::detail
