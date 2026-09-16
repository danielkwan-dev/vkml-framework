#include "hal/buffer.hpp"

#include <string>
#include <utility>

#include "hal/device.hpp"

namespace vkml::hal {

namespace {

VmaAllocationCreateFlags allocation_flags(MemoryUsage usage) {
    switch (usage) {
        case MemoryUsage::Upload:
            return VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                   VMA_ALLOCATION_CREATE_MAPPED_BIT;
        case MemoryUsage::Readback:
            return VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        case MemoryUsage::DeviceLocal: break;
    }
    return 0;
}

}  // namespace

Buffer::Buffer(const Device& device, std::uint64_t size, MemoryUsage usage)
    : allocator_(device.allocator()), size_(size) {
    const std::uint64_t limit = device.info().max_storage_buffer_range;
    if (size == 0 || size > limit) {
        throw Error("Buffer: size must be between 1 and maxStorageBufferRange (" +
                    std::to_string(limit) + ") bytes, got " + std::to_string(size));
    }

    const VkBufferCreateInfo buffer_info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                         .pNext = nullptr,
                                         .flags = 0,
                                         .size = size,
                                         .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                         .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                                         .queueFamilyIndexCount = 0,
                                         .pQueueFamilyIndices = nullptr};

    VmaAllocationCreateInfo alloc_info{};
    alloc_info.usage = usage == MemoryUsage::DeviceLocal ? VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
                                                         : VMA_MEMORY_USAGE_AUTO;
    alloc_info.flags = allocation_flags(usage);

    VmaAllocationInfo result{};
    check(vmaCreateBuffer(allocator_, &buffer_info, &alloc_info, &buffer_, &allocation_, &result),
          "vmaCreateBuffer");
    mapped_ = static_cast<std::byte*>(result.pMappedData);
}

Buffer::~Buffer() { destroy(); }

Buffer::Buffer(Buffer&& other) noexcept
    : allocator_(other.allocator_),
      buffer_(std::exchange(other.buffer_, VK_NULL_HANDLE)),
      allocation_(std::exchange(other.allocation_, VK_NULL_HANDLE)),
      size_(std::exchange(other.size_, 0)),
      mapped_(std::exchange(other.mapped_, nullptr)) {}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        destroy();
        allocator_ = other.allocator_;
        buffer_ = std::exchange(other.buffer_, VK_NULL_HANDLE);
        allocation_ = std::exchange(other.allocation_, VK_NULL_HANDLE);
        size_ = std::exchange(other.size_, 0);
        mapped_ = std::exchange(other.mapped_, nullptr);
    }
    return *this;
}

void Buffer::flush() {
    if (mapped_ != nullptr) {
        check(vmaFlushAllocation(allocator_, allocation_, 0, VK_WHOLE_SIZE), "vmaFlushAllocation");
    }
}

void Buffer::invalidate() {
    if (mapped_ != nullptr) {
        check(vmaInvalidateAllocation(allocator_, allocation_, 0, VK_WHOLE_SIZE),
              "vmaInvalidateAllocation");
    }
}

void Buffer::destroy() noexcept {
    if (buffer_ != VK_NULL_HANDLE) vmaDestroyBuffer(allocator_, buffer_, allocation_);
    buffer_ = VK_NULL_HANDLE;
    allocation_ = VK_NULL_HANDLE;
    mapped_ = nullptr;
    size_ = 0;
}

}  // namespace vkml::hal
