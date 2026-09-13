#include "vkml/context.hpp"

#include <cstdlib>

#include "core/runtime.hpp"

namespace vkml {

std::string_view to_string(DeviceType type) noexcept {
    switch (type) {
        case DeviceType::IntegratedGpu: return "integrated GPU";
        case DeviceType::DiscreteGpu: return "discrete GPU";
        case DeviceType::VirtualGpu: return "virtual GPU";
        case DeviceType::Cpu: return "CPU";
        case DeviceType::Other: break;
    }
    return "other";
}

Context::Context(ContextOptions options) {
    hal::DeviceConfig config;
    config.name_filter = std::move(options.device_name);
    if (config.name_filter.empty()) {
        if (const char* env = std::getenv("VKML_DEVICE")) config.name_filter = env;
    }
    config.enable_validation = options.enable_validation;
    runtime_ = std::make_unique<detail::Runtime>(config);
}

Context::~Context() = default;
Context::Context(Context&&) noexcept = default;
Context& Context::operator=(Context&&) noexcept = default;

const DeviceInfo& Context::device_info() const noexcept { return runtime_->device.info(); }

std::uint32_t Context::validation_error_count() const noexcept {
    return runtime_->device.validation_error_count();
}

hal::Device& Context::device() noexcept { return runtime_->device; }

}  // namespace vkml
