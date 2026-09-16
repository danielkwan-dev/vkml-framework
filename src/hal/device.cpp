#include "hal/device.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <string_view>
#include <vector>

namespace vkml::hal {

namespace {

std::string result_name(VkResult result) {
    switch (result) {
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_INCOMPLETE: return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
        case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
        case VK_ERROR_INVALID_SHADER_NV: return "VK_ERROR_INVALID_SHADER_NV";
        default: return "VkResult " + std::to_string(static_cast<int>(result));
    }
}

// volk stores device-level entry points in globals, so two live devices would
// silently share one dispatch table. Refuse the second one instead.
std::atomic<bool> g_device_alive{false};

bool contains_case_insensitive(std::string_view haystack, std::string_view needle) {
    auto lower = [](unsigned char c) { return std::tolower(c); };
    auto it = std::search(
        haystack.begin(), haystack.end(), needle.begin(), needle.end(), [&](char a, char b) {
            return lower(static_cast<unsigned char>(a)) == lower(static_cast<unsigned char>(b));
        });
    return it != haystack.end();
}

int speed_rank(VkPhysicalDeviceType type) {
    switch (type) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 4;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 3;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 2;
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return 1;
        default: return 0;
    }
}

DeviceType to_device_type(VkPhysicalDeviceType type) {
    switch (type) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return DeviceType::DiscreteGpu;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return DeviceType::IntegratedGpu;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return DeviceType::VirtualGpu;
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return DeviceType::Cpu;
        default: return DeviceType::Other;
    }
}

std::string vendor_name(std::uint32_t vendor_id) {
    switch (vendor_id) {
        case 0x1002: return "AMD";
        case 0x10DE: return "NVIDIA";
        case 0x8086: return "Intel";
        case 0x13B5: return "ARM";
        case 0x5143: return "Qualcomm";
        case 0x106B: return "Apple";
        case 0x10005: return "Mesa";
        default: {
            char buf[16];
            std::snprintf(buf, sizeof buf, "0x%04X", vendor_id);
            return buf;
        }
    }
}

std::string join_names(const std::vector<vkb::PhysicalDevice>& devices) {
    std::string out;
    for (const auto& d : devices) {
        if (!out.empty()) out += ", ";
        out += '"' + d.name + '"';
    }
    return out;
}

}  // namespace

void check(VkResult result, const char* what) {
    if (result != VK_SUCCESS) {
        throw Error(std::string(what) + " failed: " + result_name(result));
    }
}

Device::Device(const DeviceConfig& config) {
    if (g_device_alive.exchange(true)) {
        throw Error(
            "a vkml::Context is already alive in this process; only one is supported "
            "(volk keeps the device dispatch table in globals)");
    }
    try {
        create_instance(config);
        create_device(select_physical_device(config.name_filter));
        create_allocator();
    } catch (...) {
        destroy();
        g_device_alive.store(false);
        throw;
    }
}

Device::~Device() {
    destroy();
    g_device_alive.store(false);
}

void Device::create_instance(const DeviceConfig& config) {
    if (volkInitialize() != VK_SUCCESS) {
        throw Error("no Vulkan loader found; install a GPU driver with Vulkan support");
    }

    auto system = vkb::SystemInfo::get_system_info(vkGetInstanceProcAddr);
    if (!system) {
        throw Error("querying Vulkan instance support failed: " + system.error().message());
    }
    const bool validation = config.enable_validation && system->validation_layers_available;

    vkb::InstanceBuilder builder{vkGetInstanceProcAddr};
    builder.set_app_name("vkml").require_api_version(1, 2, 0).set_headless();
    if (validation) {
        builder.request_validation_layers()
            .set_debug_callback(&Device::on_validation_message)
            .set_debug_callback_user_data_pointer(this)
            .set_debug_messenger_severity(VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                          VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT);
    }

    auto instance = builder.build();
    if (!instance) {
        throw Error("creating Vulkan 1.2 instance failed: " + instance.error().message());
    }
    instance_ = *instance;
    volkLoadInstanceOnly(instance_.instance);
    info_.validation_enabled = validation;
}

vkb::PhysicalDevice Device::select_physical_device(const std::string& name_filter) const {
    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.timelineSemaphore = VK_TRUE;

    vkb::PhysicalDeviceSelector selector{instance_};
    selector.set_minimum_version(1, 2)
        .require_present(false)
        .allow_any_gpu_device_type(true)
        .set_required_features_12(features12);

    auto candidates = selector.select_devices();
    if (!candidates || candidates->empty()) {
        throw Error("no GPU supports Vulkan 1.2 compute with timeline semaphores");
    }

    if (!name_filter.empty()) {
        for (const auto& device : *candidates) {
            if (contains_case_insensitive(device.name, name_filter)) return device;
        }
        throw Error("no suitable device name contains \"" + name_filter +
                    "\"; available: " + join_names(*candidates));
    }

    // max_element returns the first of equals, so ties keep the driver's order
    return *std::ranges::max_element(*candidates, {}, [](const vkb::PhysicalDevice& d) {
        return speed_rank(d.properties.deviceType);
    });
}

void Device::create_device(const vkb::PhysicalDevice& physical) {
    auto device = vkb::DeviceBuilder{physical}.build();
    if (!device) {
        throw Error("creating Vulkan device on " + physical.name +
                    " failed: " + device.error().message());
    }
    device_ = *device;
    volkLoadDevice(device_.device);

    // vk-bootstrap creates one queue per family. Any compute-capable family
    // works; picking a dedicated async-compute family is a later optimization.
    const auto& families = device_.queue_families;
    auto compute = std::ranges::find_if(families, [](const VkQueueFamilyProperties& f) {
        return (f.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
    });
    if (compute == families.end()) {
        throw Error(physical.name + " has no compute-capable queue family");
    }
    queue_family_ = static_cast<std::uint32_t>(compute - families.begin());
    vkGetDeviceQueue(device_.device, queue_family_, 0, &queue_);

    query_info();
}

void Device::create_allocator() {
    VmaVulkanFunctions functions{};
    functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo create_info{};
    create_info.vulkanApiVersion = VK_API_VERSION_1_2;
    create_info.instance = instance_.instance;
    create_info.physicalDevice = physical_device();
    create_info.device = device_.device;
    create_info.pVulkanFunctions = &functions;
    check(vmaCreateAllocator(&create_info, &allocator_), "vmaCreateAllocator");
}

void Device::query_info() {
    VkPhysicalDeviceVulkan12Properties props12{};
    props12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES;
    VkPhysicalDeviceSubgroupProperties subgroup{};
    subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    subgroup.pNext = &props12;
    VkPhysicalDeviceProperties2 props{};
    props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props.pNext = &subgroup;
    vkGetPhysicalDeviceProperties2(physical_device(), &props);

    const VkPhysicalDeviceProperties& p = props.properties;
    const VkPhysicalDeviceLimits& limits = p.limits;

    DeviceInfo info;
    info.name = p.deviceName;
    info.vendor = vendor_name(p.vendorID);
    info.driver = std::string(props12.driverName) + " " + props12.driverInfo;
    info.type = to_device_type(p.deviceType);
    info.api_version = {VK_API_VERSION_MAJOR(p.apiVersion), VK_API_VERSION_MINOR(p.apiVersion),
                        VK_API_VERSION_PATCH(p.apiVersion)};

    info.unified_memory = info.type == DeviceType::IntegratedGpu || info.type == DeviceType::Cpu;
    const VkPhysicalDeviceMemoryProperties& memory = device_.physical_device.memory_properties;
    for (std::uint32_t i = 0; i < memory.memoryHeapCount; ++i) {
        if (memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            info.device_local_bytes = std::max(info.device_local_bytes, memory.memoryHeaps[i].size);
        }
    }

    info.subgroup_size = subgroup.subgroupSize;
    info.max_workgroup_invocations = limits.maxComputeWorkGroupInvocations;
    std::ranges::copy(limits.maxComputeWorkGroupSize, info.max_workgroup_size.begin());
    std::ranges::copy(limits.maxComputeWorkGroupCount, info.max_workgroup_count.begin());
    info.max_shared_memory_bytes = limits.maxComputeSharedMemorySize;
    info.max_push_constant_bytes = limits.maxPushConstantsSize;

    info.max_storage_buffer_range = limits.maxStorageBufferRange;
    info.min_storage_buffer_offset_alignment = limits.minStorageBufferOffsetAlignment;
    info.non_coherent_atom_size = limits.nonCoherentAtomSize;
    info.validation_enabled = info_.validation_enabled;  // decided in create_instance

    info_ = std::move(info);
}

void Device::destroy() noexcept {
    if (allocator_ != VK_NULL_HANDLE) {
        vmaDestroyAllocator(allocator_);
        allocator_ = VK_NULL_HANDLE;
    }
    if (device_.device != VK_NULL_HANDLE) {
        vkb::destroy_device(device_);
        device_ = {};
    }
    if (instance_.instance != VK_NULL_HANDLE) {
        vkb::destroy_instance(instance_);  // also destroys the debug messenger
        instance_ = {};
    }
}

VKAPI_ATTR VkBool32 VKAPI_CALL Device::on_validation_message(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT /*types*/,
    const VkDebugUtilsMessengerCallbackDataEXT* data, void* user_data) {
    const bool is_error = severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    if (is_error) {
        static_cast<Device*>(user_data)->validation_errors_.fetch_add(1, std::memory_order_relaxed);
    }
    std::fprintf(stderr, "[vkml validation %s] %s\n", is_error ? "error" : "warning",
                 data && data->pMessage ? data->pMessage : "(no message)");
    return VK_FALSE;  // never abort the call; the count is what tests check
}

}  // namespace vkml::hal
