#pragma once

#include <cstdint>
#include <span>

#include "hal/vulkan.hpp"

namespace vkml::hal {

class Device;

inline constexpr std::uint32_t kSpirvMagic = 0x07230203u;

// A compute pipeline plus its layouts. The shader's bindings must be storage
// buffers 0..storage_buffer_count-1 in set 0, and its push-constant block at
// most push_constant_bytes. Specialization constants are supplied in
// constant_id order 0, 1, 2, ..., each a 32-bit value.
class ComputePipeline {
public:
    ComputePipeline(const Device& device, std::span<const std::uint32_t> spirv,
                    std::uint32_t storage_buffer_count, std::uint32_t push_constant_bytes,
                    std::span<const std::uint32_t> specialization = {});
    ~ComputePipeline();

    ComputePipeline(const ComputePipeline&) = delete;
    ComputePipeline& operator=(const ComputePipeline&) = delete;

    VkPipeline pipeline() const noexcept { return pipeline_; }
    VkPipelineLayout layout() const noexcept { return layout_; }
    VkDescriptorSetLayout set_layout() const noexcept { return set_layout_; }

private:
    void destroy() noexcept;

    VkDevice device_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

}  // namespace vkml::hal
