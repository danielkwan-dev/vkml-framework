#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "vkml/error.hpp"

namespace vkml {

namespace hal {
class Device;
}  // namespace hal

namespace detail {
class Runtime;
}  // namespace detail

enum class DeviceType : std::uint8_t { Other, IntegratedGpu, DiscreteGpu, VirtualGpu, Cpu };

std::string_view to_string(DeviceType type) noexcept;

// Everything about the selected GPU that kernels and users need to make
// decisions. Plain C++ types only: no Vulkan headers leak through the public API.
struct DeviceInfo {
    struct Version {
        std::uint32_t major = 0, minor = 0, patch = 0;
    };

    std::string name;
    std::string vendor;
    std::string driver;
    DeviceType type = DeviceType::Other;
    Version api_version;

    // Integrated GPUs and CPU implementations share memory with the host, so
    // uploads can skip the staging copy a discrete GPU needs.
    bool unified_memory = false;
    std::uint64_t device_local_bytes = 0;  // largest device-local heap

    // Compute limits. Tile and workgroup sizes are derived from these, never
    // hard-coded: the spec only guarantees 128 invocations and 16 KiB shared memory.
    std::uint32_t subgroup_size = 0;
    std::uint32_t max_workgroup_invocations = 0;
    std::array<std::uint32_t, 3> max_workgroup_size{};
    std::array<std::uint32_t, 3> max_workgroup_count{};
    std::uint32_t max_shared_memory_bytes = 0;
    std::uint32_t max_push_constant_bytes = 0;

    // Buffer limits and alignment rules for suballocation and mapped memory.
    std::uint64_t max_storage_buffer_range = 0;
    std::uint64_t min_storage_buffer_offset_alignment = 0;
    std::uint64_t non_coherent_atom_size = 0;

    bool validation_enabled = false;

    // GPU timestamps, for Context::set_profiling: whether the compute queue
    // records them, and nanoseconds per timestamp tick.
    bool timestamps = false;
    double timestamp_period_ns = 0.0;
};

// Time the GPU spent in one kernel (or in copies, as "copy"), from
// Context::profile.
struct KernelTime {
    std::string name;
    std::uint64_t calls = 0;
    double milliseconds = 0.0;
};

struct ContextOptions {
    // Case-insensitive substring of the device name to use. When empty, the
    // VKML_DEVICE environment variable is consulted, and then the fastest
    // device type wins (discrete > integrated > virtual > CPU).
    std::string device_name;

    // Khronos validation layers, if installed. On by default in debug builds.
#ifdef NDEBUG
    bool enable_validation = false;
#else
    bool enable_validation = true;
#endif
};

// Owns the Vulkan instance, device, compute queue and memory allocator.
// Create one and keep it alive for as long as any GPU work is in use.
//
// Only one Context may be alive per process for now: volk keeps the device
// dispatch table in globals. Constructing a second throws vkml::Error.
class Context {
public:
    explicit Context(ContextOptions options = {});
    ~Context();

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) noexcept;
    // A moved-from Context may only be destroyed or assigned.
    Context& operator=(Context&&) noexcept;

    const DeviceInfo& device_info() const noexcept;

    // Validation-layer errors reported since creation. Tests assert this stays 0.
    std::uint32_t validation_error_count() const noexcept;

    // Profiling times every kernel with GPU timestamps; it slows execution a
    // little. profile() covers work that has finished and been waited for
    // (reading a tensor back waits), slowest kernel first. Throws if the
    // device has no timestamps (DeviceInfo::timestamps).
    void set_profiling(bool enabled);
    std::vector<KernelTime> profile() const;
    void reset_profile();

    // Backend access for op implementations; not needed by end users.
    hal::Device& device() noexcept;
    detail::Runtime& runtime() noexcept { return *runtime_; }

private:
    std::unique_ptr<detail::Runtime> runtime_;
};

}  // namespace vkml
