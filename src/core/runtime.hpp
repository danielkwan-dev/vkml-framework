#pragma once

#include <cstdint>
#include <initializer_list>
#include <map>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "hal/buffer.hpp"
#include "hal/device.hpp"
#include "hal/pipeline.hpp"
#include "hal/stream.hpp"

namespace vkml::detail {

// Everything a Context owns. Lives on the heap so tensors can point at it and
// survive the Context being moved.
//
// Kernels and the stream are declared after the device, so they are destroyed
// before it.
//
// Not thread-safe: tensors of one Context must be used from one thread.
class Runtime {
public:
    explicit Runtime(const hal::DeviceConfig& config);
    ~Runtime();  // finishes all recorded work, so no retired buffer is still in use

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    hal::Device device;
    hal::Stream stream;

    // Takes a buffer that recorded or submitted work may still use and frees
    // it once the stream has finished that work. When retired buffers hold
    // more than the retired limit, waits for the work and frees them all, so
    // long runs of recorded work (loading a model's weights through staging
    // buffers, say) cannot hold unbounded memory.
    void retire(hal::Buffer buffer);
    std::uint64_t retired_bytes() const noexcept { return retired_bytes_; }
    void set_retired_limit(std::uint64_t bytes) noexcept { retired_limit_ = bytes; }

    // A buffer of exactly size bytes, reused from the pool of freed buffers
    // whose work has finished when one fits, else newly created. Creating
    // buffers gets slower as a process holds more allocations (drivers track
    // them for every submission), and a model's forward pass frees and
    // allocates hundreds of same-sized buffers per token.
    hal::Buffer acquire(std::uint64_t size, hal::MemoryUsage usage);
    std::uint64_t pooled_bytes() const noexcept { return pooled_bytes_; }
    void set_pool_limit(std::uint64_t bytes);

    // Waits for all recorded work and frees every retired buffer.
    void synchronize();

    // Workgroup width for 1-D kernels, from the device limits.
    std::uint32_t workgroup_width() const noexcept;
    // Workgroup count for a grid-stride kernel over count items.
    std::uint32_t workgroup_count(std::uint64_t count) const noexcept;

    // Records a fill of every word of buffer with zero bits.
    void fill_zeros(const hal::Buffer& buffer);

    // The pipeline for a kernel, compiled on first use and cached by
    // (kernel, specialization constants). label names it in profiles.
    const hal::ComputePipeline& pipeline(std::string_view label,
                                         std::span<const std::uint32_t> spirv,
                                         std::uint32_t storage_buffer_count,
                                         std::uint32_t push_constant_bytes,
                                         std::initializer_list<std::uint32_t> specialization);

    // Side of the square matmul workgroup: 16 where the device allows 256
    // invocations, else 8 (the spec only guarantees 128).
    std::uint32_t matmul_tile() const noexcept;

private:
    void collect();
    void release(hal::Buffer buffer);  // into the pool, or destroyed past its limit

    std::vector<std::pair<std::uint64_t, hal::Buffer>> retired_;  // (stream value, buffer)
    std::uint64_t retired_bytes_ = 0;
    std::uint64_t retired_limit_ = std::uint64_t{256} << 20;
    std::multimap<std::pair<hal::MemoryUsage, std::uint64_t>, hal::Buffer> pool_;
    std::uint64_t pooled_bytes_ = 0;
    std::uint64_t pool_limit_ = std::uint64_t{512} << 20;
    using PipelineKey = std::pair<const std::uint32_t*, std::vector<std::uint32_t>>;
    std::map<PipelineKey, std::unique_ptr<hal::ComputePipeline>> pipelines_;
};

}  // namespace vkml::detail
