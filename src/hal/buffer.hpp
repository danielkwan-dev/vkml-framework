#pragma once

#include <cstddef>
#include <cstdint>

#include "hal/vulkan.hpp"

namespace vkml::hal {

class Device;

// Where a buffer lives, which decides how the host can reach it.
enum class MemoryUsage : std::uint8_t {
    DeviceLocal,  // fastest for kernels; the host reaches it only through copies
    Upload,       // host-visible, written sequentially by the host, read by the GPU
    Readback,     // host-visible and cached, written by the GPU, read by the host
};

// A storage buffer and its VMA allocation. Every buffer can be bound to a
// kernel and used as either end of a copy.
class Buffer {
public:
    Buffer(const Device& device, std::uint64_t size, MemoryUsage usage);
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;  // a moved-from Buffer has a null handle

    VkBuffer handle() const noexcept { return buffer_; }
    std::uint64_t size() const noexcept { return size_; }

    // Persistently mapped for Upload and Readback buffers; null for DeviceLocal.
    std::byte* mapped() const noexcept { return mapped_; }
    MemoryUsage usage() const noexcept { return usage_; }

    // Make host writes visible to the GPU (flush) or GPU writes visible to the
    // host (invalidate). No-ops on coherent memory, which most drivers pick.
    void flush();
    void invalidate();

private:
    void destroy() noexcept;

    VmaAllocator allocator_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    std::uint64_t size_ = 0;
    std::byte* mapped_ = nullptr;
    MemoryUsage usage_ = MemoryUsage::DeviceLocal;
};

}  // namespace vkml::hal
