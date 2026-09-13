#pragma once

#include <cstdint>
#include <memory>

#include "hal/device.hpp"
#include "hal/pipeline.hpp"
#include "hal/stream.hpp"

namespace vkml::detail {

// Everything a Context owns. Lives on the heap so tensors can point at it and
// survive the Context being moved. Members are destroyed in reverse order, so
// kernels and the stream go before the device they were created on.
//
// Not thread-safe: tensors of one Context must be used from one thread.
class Runtime {
public:
    explicit Runtime(const hal::DeviceConfig& config);

    hal::Device device;
    hal::Stream stream;

    // Workgroup width for 1-D kernels, from the device limits.
    std::uint32_t workgroup_width() const noexcept;

    // Kernels are compiled into pipelines on first use.
    const hal::ComputePipeline& fill();

private:
    std::unique_ptr<hal::ComputePipeline> fill_;
};

}  // namespace vkml::detail
