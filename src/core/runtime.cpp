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

std::uint32_t Runtime::workgroup_width() const noexcept {
    const DeviceInfo& d = device.info();
    return std::min({256u, d.max_workgroup_size[0], d.max_workgroup_invocations});
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

}  // namespace vkml::detail
