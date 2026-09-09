#pragma once

#include <atomic>
#include <cstdint>
#include <string>

#include "hal/vulkan.hpp"
#include "vkml/context.hpp"

namespace vkml::hal {

// Throws vkml::Error naming the call and the VkResult unless result is VK_SUCCESS.
void check(VkResult result, const char* what);

struct DeviceConfig {
    std::string name_filter;  // case-insensitive substring; empty ranks by device type
    bool enable_validation = false;
};

// Instance, physical device, logical device, one compute queue and the VMA
// allocator. Requires Vulkan 1.2 and timeline semaphores, which every
// desktop driver and lavapipe provide.
class Device {
public:
    explicit Device(const DeviceConfig& config);
    ~Device();

    // Not movable: the validation callback holds a pointer to this object.
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    VkInstance instance() const noexcept { return instance_.instance; }
    VkPhysicalDevice physical_device() const noexcept { return device_.physical_device.physical_device; }
    VkDevice device() const noexcept { return device_.device; }
    VkQueue compute_queue() const noexcept { return queue_; }
    std::uint32_t compute_queue_family() const noexcept { return queue_family_; }
    VmaAllocator allocator() const noexcept { return allocator_; }

    const DeviceInfo& info() const noexcept { return info_; }
    std::uint32_t validation_error_count() const noexcept {
        return validation_errors_.load(std::memory_order_relaxed);
    }

private:
    static VKAPI_ATTR VkBool32 VKAPI_CALL on_validation_message(
        VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT types,
        const VkDebugUtilsMessengerCallbackDataEXT* data, void* user_data);

    void create_instance(const DeviceConfig& config);
    vkb::PhysicalDevice select_physical_device(const std::string& name_filter) const;
    void create_device(const vkb::PhysicalDevice& physical);
    void create_allocator();
    void query_info();
    void destroy() noexcept;

    vkb::Instance instance_;
    vkb::Device device_;
    VkQueue queue_ = VK_NULL_HANDLE;
    std::uint32_t queue_family_ = 0;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    DeviceInfo info_;
    std::atomic<std::uint32_t> validation_errors_{0};
};

}  // namespace vkml::hal
