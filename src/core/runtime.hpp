#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "hal/buffer.hpp"
#include "hal/device.hpp"
#include "hal/pipeline.hpp"
#include "hal/stream.hpp"

namespace vkml::detail {

enum class BinaryOp : std::uint32_t { Add, Sub, Mul, Div };  // matches shaders/binary.comp
enum class UnaryOp : std::uint32_t { Silu, Gelu };            // matches shaders/unary.comp

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
    // it once the stream has finished that work.
    void retire(hal::Buffer buffer);

    // Waits for all recorded work and frees every retired buffer.
    void synchronize();

    // Workgroup width for 1-D kernels, from the device limits.
    std::uint32_t workgroup_width() const noexcept;
    // Workgroup count for a grid-stride kernel over count items.
    std::uint32_t workgroup_count(std::uint64_t count) const noexcept;

    // Records a fill of every word of buffer with zero bits.
    void fill_zeros(const hal::Buffer& buffer);

    // Kernels are compiled into pipelines on first use.
    const hal::ComputePipeline& fill();
    const hal::ComputePipeline& binary(BinaryOp op);
    const hal::ComputePipeline& unary(UnaryOp op);
    const hal::ComputePipeline& matmul();

    // Side of the square matmul workgroup: 16 where the device allows 256
    // invocations, else 8 (the spec only guarantees 128).
    std::uint32_t matmul_tile() const noexcept;

private:
    void collect();

    std::vector<std::pair<std::uint64_t, hal::Buffer>> retired_;  // (stream value, buffer)
    std::unique_ptr<hal::ComputePipeline> fill_;
    std::array<std::unique_ptr<hal::ComputePipeline>, 4> binary_;
    std::array<std::unique_ptr<hal::ComputePipeline>, 2> unary_;
    std::unique_ptr<hal::ComputePipeline> matmul_;
};

}  // namespace vkml::detail
