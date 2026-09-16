#include "hal/stream.hpp"

#include <limits>
#include <string>

#include "hal/buffer.hpp"
#include "hal/device.hpp"
#include "hal/pipeline.hpp"

namespace vkml::hal {

namespace {

// Sized for a typical submission; more pools are added when one fills up.
constexpr std::uint32_t kSetsPerPool = 256;
constexpr std::uint32_t kBuffersPerPool = 1024;

constexpr VkPipelineStageFlags kWorkStages =
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
constexpr VkAccessFlags kWorkWrites = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
constexpr VkAccessFlags kWorkAccess = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                      VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;

void memory_barrier(VkCommandBuffer cmd, VkPipelineStageFlags dst_stages,
                    VkAccessFlags dst_access) {
    const VkMemoryBarrier barrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                  .pNext = nullptr,
                                  .srcAccessMask = kWorkWrites,
                                  .dstAccessMask = dst_access};
    vkCmdPipelineBarrier(cmd, kWorkStages, dst_stages, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

}  // namespace

Stream::Stream(const Device& device) : device_(&device) {
    const VkDevice vk = device.device();
    try {
        const VkCommandPoolCreateInfo pool_info{.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                                .pNext = nullptr,
                                                .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                                                .queueFamilyIndex = device.compute_queue_family()};
        check(vkCreateCommandPool(vk, &pool_info, nullptr, &command_pool_), "vkCreateCommandPool");

        const VkCommandBufferAllocateInfo cmd_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = command_pool_,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1};
        check(vkAllocateCommandBuffers(vk, &cmd_info, &command_buffer_),
              "vkAllocateCommandBuffers");

        const VkSemaphoreTypeCreateInfo type_info{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
            .pNext = nullptr,
            .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
            .initialValue = 0};
        const VkSemaphoreCreateInfo semaphore_info{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &type_info, .flags = 0};
        check(vkCreateSemaphore(vk, &semaphore_info, nullptr, &timeline_), "vkCreateSemaphore");
    } catch (...) {
        destroy();
        throw;
    }
}

Stream::~Stream() {
    try {
        wait(submitted_);
    } catch (...) {
        // Device lost: nothing is running any more, so destroying is still safe.
    }
    destroy();
}

void Stream::dispatch(const ComputePipeline& pipeline, std::span<const Buffer* const> buffers,
                      std::span<const std::byte> push_constants,
                      std::array<std::uint32_t, 3> groups) {
    if (buffers.size() != pipeline.storage_buffer_count()) {
        throw Error("Stream::dispatch: pipeline takes " +
                    std::to_string(pipeline.storage_buffer_count()) + " storage buffers, got " +
                    std::to_string(buffers.size()));
    }
    if (push_constants.size() != pipeline.push_constant_bytes()) {
        throw Error("Stream::dispatch: pipeline takes " +
                    std::to_string(pipeline.push_constant_bytes()) +
                    " bytes of push constants, got " + std::to_string(push_constants.size()));
    }
    const auto& max_groups = device_->info().max_workgroup_count;
    for (std::size_t i = 0; i < 3; ++i) {
        if (groups[i] == 0 || groups[i] > max_groups[i]) {
            throw Error("Stream::dispatch: workgroup count " + std::to_string(groups[i]) +
                        " in dimension " + std::to_string(i) +
                        " must be between 1 and maxComputeWorkGroupCount (" +
                        std::to_string(max_groups[i]) + ")");
        }
    }

    const VkCommandBuffer cmd = begin_command();
    const VkDescriptorSet set = allocate_descriptor_set(pipeline.set_layout());

    std::vector<VkDescriptorBufferInfo> infos(buffers.size());
    std::vector<VkWriteDescriptorSet> writes(buffers.size());
    for (std::size_t i = 0; i < buffers.size(); ++i) {
        infos[i] = {.buffer = buffers[i]->handle(), .offset = 0, .range = VK_WHOLE_SIZE};
        writes[i] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                     .pNext = nullptr,
                     .dstSet = set,
                     .dstBinding = static_cast<std::uint32_t>(i),
                     .dstArrayElement = 0,
                     .descriptorCount = 1,
                     .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                     .pImageInfo = nullptr,
                     .pBufferInfo = &infos[i],
                     .pTexelBufferView = nullptr};
    }
    vkUpdateDescriptorSets(device_->device(), static_cast<std::uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout(), 0, 1, &set, 0,
                            nullptr);
    if (!push_constants.empty()) {
        vkCmdPushConstants(cmd, pipeline.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           static_cast<std::uint32_t>(push_constants.size()),
                           push_constants.data());
    }
    vkCmdDispatch(cmd, groups[0], groups[1], groups[2]);
}

void Stream::copy(const Buffer& src, const Buffer& dst, std::uint64_t bytes) {
    if (bytes == 0 || bytes > src.size() || bytes > dst.size()) {
        throw Error("Stream::copy: cannot copy " + std::to_string(bytes) + " bytes from a " +
                    std::to_string(src.size()) + "-byte buffer to a " + std::to_string(dst.size()) +
                    "-byte buffer");
    }
    const VkBufferCopy region{.srcOffset = 0, .dstOffset = 0, .size = bytes};
    vkCmdCopyBuffer(begin_command(), src.handle(), dst.handle(), 1, &region);
}

std::uint64_t Stream::submit() {
    if (!recording_) return submitted_;

    // Make every write available to the host once the timeline signals.
    memory_barrier(command_buffer_, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
    recording_ = false;
    check(vkEndCommandBuffer(command_buffer_), "vkEndCommandBuffer");

    const std::uint64_t value = submitted_ + 1;
    const VkTimelineSemaphoreSubmitInfo timeline_info{
        .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreValueCount = 0,
        .pWaitSemaphoreValues = nullptr,
        .signalSemaphoreValueCount = 1,
        .pSignalSemaphoreValues = &value};
    const VkSubmitInfo submit_info{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                   .pNext = &timeline_info,
                                   .waitSemaphoreCount = 0,
                                   .pWaitSemaphores = nullptr,
                                   .pWaitDstStageMask = nullptr,
                                   .commandBufferCount = 1,
                                   .pCommandBuffers = &command_buffer_,
                                   .signalSemaphoreCount = 1,
                                   .pSignalSemaphores = &timeline_};
    check(vkQueueSubmit(device_->compute_queue(), 1, &submit_info, VK_NULL_HANDLE),
          "vkQueueSubmit");
    submitted_ = value;
    return value;
}

void Stream::wait(std::uint64_t value) {
    if (value > submitted_) {
        throw Error("Stream::wait: value " + std::to_string(value) +
                    " has not been submitted (last is " + std::to_string(submitted_) + ")");
    }
    const VkSemaphoreWaitInfo info{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                                   .pNext = nullptr,
                                   .flags = 0,
                                   .semaphoreCount = 1,
                                   .pSemaphores = &timeline_,
                                   .pValues = &value};
    check(vkWaitSemaphores(device_->device(), &info, std::numeric_limits<std::uint64_t>::max()),
          "vkWaitSemaphores");
}

std::uint64_t Stream::completed() const {
    std::uint64_t value = 0;
    check(vkGetSemaphoreCounterValue(device_->device(), timeline_, &value),
          "vkGetSemaphoreCounterValue");
    return value;
}

VkCommandBuffer Stream::begin_command() {
    if (!recording_) {
        // The command buffer and descriptor sets may still be in use by the
        // previous submission.
        wait(submitted_);
        const VkDevice vk = device_->device();
        check(vkResetCommandPool(vk, command_pool_, 0), "vkResetCommandPool");
        for (VkDescriptorPool pool : descriptor_pools_) {
            check(vkResetDescriptorPool(vk, pool, 0), "vkResetDescriptorPool");
        }
        current_pool_ = 0;

        const VkCommandBufferBeginInfo begin_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr};
        check(vkBeginCommandBuffer(command_buffer_, &begin_info), "vkBeginCommandBuffer");
        recording_ = true;
        empty_ = true;
    }
    if (!empty_) memory_barrier(command_buffer_, kWorkStages, kWorkAccess);
    empty_ = false;
    return command_buffer_;
}

VkDescriptorSet Stream::allocate_descriptor_set(VkDescriptorSetLayout layout) {
    const VkDevice vk = device_->device();
    for (;;) {
        if (current_pool_ == descriptor_pools_.size()) {
            const VkDescriptorPoolSize size{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            .descriptorCount = kBuffersPerPool};
            const VkDescriptorPoolCreateInfo info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .maxSets = kSetsPerPool,
                .poolSizeCount = 1,
                .pPoolSizes = &size};
            VkDescriptorPool pool = VK_NULL_HANDLE;
            check(vkCreateDescriptorPool(vk, &info, nullptr, &pool), "vkCreateDescriptorPool");
            descriptor_pools_.push_back(pool);
        }

        const VkDescriptorSetAllocateInfo info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = descriptor_pools_[current_pool_],
            .descriptorSetCount = 1,
            .pSetLayouts = &layout};
        VkDescriptorSet set = VK_NULL_HANDLE;
        const VkResult result = vkAllocateDescriptorSets(vk, &info, &set);
        if (result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTED_POOL) {
            ++current_pool_;
            continue;
        }
        check(result, "vkAllocateDescriptorSets");
        return set;
    }
}

void Stream::destroy() noexcept {
    const VkDevice vk = device_->device();
    for (VkDescriptorPool pool : descriptor_pools_) vkDestroyDescriptorPool(vk, pool, nullptr);
    descriptor_pools_.clear();
    if (command_pool_ != VK_NULL_HANDLE) vkDestroyCommandPool(vk, command_pool_, nullptr);
    if (timeline_ != VK_NULL_HANDLE) vkDestroySemaphore(vk, timeline_, nullptr);
    command_pool_ = VK_NULL_HANDLE;
    command_buffer_ = VK_NULL_HANDLE;
    timeline_ = VK_NULL_HANDLE;
}

}  // namespace vkml::hal
