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
    retired_bytes_ += buffer.size();
    retired_.emplace_back(stream.pending_value(), std::move(buffer));
    // Buffers retired while recording wait on work not yet submitted, which
    // cannot have finished: only look (a driver call and a scan) when the
    // oldest retired buffer belongs to a submission already made.
    if (retired_.front().first <= stream.submitted()) collect();
    if (retired_bytes_ > retired_limit_) synchronize();
}

void Runtime::synchronize() {
    stream.synchronize();
    for (auto& [value, buffer] : retired_) release(std::move(buffer));
    retired_.clear();
    retired_bytes_ = 0;
}

hal::Buffer Runtime::acquire(std::uint64_t size, hal::MemoryUsage usage) {
    // Small sizes are rounded up to 4 KiB, so tensors that differ by a few
    // bytes, such as attention scores that grow a key per token, share pooled
    // buffers. Large ones (weights, caches) stay exact to waste nothing.
    constexpr std::uint64_t kSmall = std::uint64_t{1} << 20, kGranule = 4096;
    const std::uint64_t rounded = (size + kGranule - 1) / kGranule * kGranule;
    if (size <= kSmall && rounded <= device.info().max_storage_buffer_range) size = rounded;

    if (const auto it = pool_.find({usage, size}); it != pool_.end()) {
        hal::Buffer buffer = std::move(it->second);
        pool_.erase(it);
        pooled_bytes_ -= size;
        return buffer;
    }
    return hal::Buffer{device, size, usage};
}

void Runtime::release(hal::Buffer buffer) {
    if (buffer.handle() == VK_NULL_HANDLE || pooled_bytes_ + buffer.size() > pool_limit_) return;
    pooled_bytes_ += buffer.size();
    const auto key = std::pair{buffer.usage(), buffer.size()};
    pool_.emplace(key, std::move(buffer));
}

void Runtime::set_pool_limit(std::uint64_t bytes) {
    pool_limit_ = bytes;
    while (pooled_bytes_ > pool_limit_ && !pool_.empty()) {
        pooled_bytes_ -= pool_.begin()->second.size();
        pool_.erase(pool_.begin());
    }
}

void Runtime::collect() {
    // Buffers are retired in submission order, so the finished ones lead.
    const std::uint64_t done = stream.completed();
    std::size_t finished = 0;
    for (; finished < retired_.size() && retired_[finished].first <= done; ++finished) {
        retired_bytes_ -= retired_[finished].second.size();
        release(std::move(retired_[finished].second));
    }
    retired_.erase(retired_.begin(), retired_.begin() + std::ptrdiff_t(finished));
}

void Runtime::fill_zeros(const hal::Buffer& buffer) {
    const auto words = static_cast<std::uint32_t>(buffer.size() / 4);
    const FillParams params{words, 0u};
    const std::array<const hal::Buffer*, 1> buffers{&buffer};
    stream.dispatch(pipeline("fill", shaders::fill, 1, sizeof(FillParams), {workgroup_width()}),
                    buffers, std::as_bytes(std::span{&params, 1}), {workgroup_count(words), 1, 1});
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

const hal::ComputePipeline& Runtime::pipeline(std::string_view label,
                                              std::span<const std::uint32_t> spirv,
                                              std::uint32_t storage_buffer_count,
                                              std::uint32_t push_constant_bytes,
                                              std::initializer_list<std::uint32_t> specialization) {
    // Kernels are static arrays, so the address identifies the kernel.
    auto& slot = pipelines_[PipelineKey{spirv.data(), specialization}];
    if (!slot) {
        const std::vector<std::uint32_t> spec{specialization};
        slot = std::make_unique<hal::ComputePipeline>(
            device, spirv, storage_buffer_count, push_constant_bytes, spec, std::string(label));
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
