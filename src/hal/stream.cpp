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

// Timestamp queries per submission when profiling: a start and an end for up
// to 4096 commands. Commands beyond that go untimed.
constexpr std::uint32_t kQueryCapacity = 8192;

constexpr VkPipelineStageFlags kWorkStages =
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
constexpr VkAccessFlags kWorkWrites = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
constexpr VkAccessFlags kWorkAccess = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                      VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;

// A barrier's first scope is everything earlier in submission order on the
// queue, so one at the start of a command buffer also orders it after earlier
// submissions that may still be running.
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
        for (Slot& s : slots_) {
            const VkCommandPoolCreateInfo pool_info{
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                .queueFamilyIndex = device.compute_queue_family()};
            check(vkCreateCommandPool(vk, &pool_info, nullptr, &s.command_pool),
                  "vkCreateCommandPool");
            const VkCommandBufferAllocateInfo cmd_info{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .pNext = nullptr,
                .commandPool = s.command_pool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1};
            check(vkAllocateCommandBuffers(vk, &cmd_info, &s.command_buffer),
                  "vkAllocateCommandBuffers");
        }

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
    begin_timestamp(cmd, pipeline.label().empty() ? std::string("unnamed") : pipeline.label());
    vkCmdDispatch(cmd, groups[0], groups[1], groups[2]);
    end_timestamp(cmd);
}

void Stream::copy(const Buffer& src, const Buffer& dst, std::uint64_t bytes) {
    if (bytes == 0 || bytes > src.size() || bytes > dst.size()) {
        throw Error("Stream::copy: cannot copy " + std::to_string(bytes) + " bytes from a " +
                    std::to_string(src.size()) + "-byte buffer to a " + std::to_string(dst.size()) +
                    "-byte buffer");
    }
    const VkBufferCopy region{.srcOffset = 0, .dstOffset = 0, .size = bytes};
    const VkCommandBuffer cmd = begin_command();
    begin_timestamp(cmd, "copy");
    vkCmdCopyBuffer(cmd, src.handle(), dst.handle(), 1, &region);
    end_timestamp(cmd);
}

std::uint64_t Stream::submit() {
    if (!recording_) return submitted_;
    Slot& s = slot();

    // Make every write available to the host once the timeline signals.
    memory_barrier(s.command_buffer, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
    recording_ = false;
    check(vkEndCommandBuffer(s.command_buffer), "vkEndCommandBuffer");

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
                                   .pCommandBuffers = &s.command_buffer,
                                   .signalSemaphoreCount = 1,
                                   .pSignalSemaphores = &timeline_};
    check(vkQueueSubmit(device_->compute_queue(), 1, &submit_info, VK_NULL_HANDLE),
          "vkQueueSubmit");
    submitted_ = value;
    s.value = value;
    s.timestamps_pending = !s.query_labels.empty();
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
    for (Slot& s : slots_) {
        if (s.timestamps_pending && s.value <= value) collect_timestamps(s);
    }
}

std::uint64_t Stream::completed() const {
    std::uint64_t value = 0;
    check(vkGetSemaphoreCounterValue(device_->device(), timeline_, &value),
          "vkGetSemaphoreCounterValue");
    return value;
}

VkCommandBuffer Stream::begin_command() {
    if (!recording_) {
        // Move to the next slot; its command buffer and descriptor sets may
        // still be in use by the submission that last took it.
        current_ = (current_ + 1) % kSlots;
        Slot& s = slot();
        wait(s.value);
        const VkDevice vk = device_->device();
        check(vkResetCommandPool(vk, s.command_pool, 0), "vkResetCommandPool");
        for (VkDescriptorPool pool : s.descriptor_pools) {
            check(vkResetDescriptorPool(vk, pool, 0), "vkResetDescriptorPool");
        }
        s.current_pool = 0;

        const VkCommandBufferBeginInfo begin_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr};
        check(vkBeginCommandBuffer(s.command_buffer, &begin_info), "vkBeginCommandBuffer");
        if (profiling_) {
            if (s.query_pool == VK_NULL_HANDLE) {
                const VkQueryPoolCreateInfo info{.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                                 .pNext = nullptr,
                                                 .flags = 0,
                                                 .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                                 .queryCount = kQueryCapacity,
                                                 .pipelineStatistics = 0};
                check(vkCreateQueryPool(vk, &info, nullptr, &s.query_pool), "vkCreateQueryPool");
            }
            vkCmdResetQueryPool(s.command_buffer, s.query_pool, 0, kQueryCapacity);
        }
        s.query_labels.clear();
        recording_ = true;
        // Earlier submissions may still be running: the first command follows them too.
        barrier_needed_ = submitted_ > 0;
    }
    const VkCommandBuffer cmd = slot().command_buffer;
    if (barrier_needed_) memory_barrier(cmd, kWorkStages, kWorkAccess);
    barrier_needed_ = true;
    return cmd;
}

void Stream::set_profiling(bool enabled) {
    if (enabled == profiling_) return;
    synchronize();  // recorded work runs, and is timed, under the old setting
    if (enabled && !device_->info().timestamps) {
        throw Error("Stream::set_profiling: " + device_->info().name +
                    " does not record timestamps on its compute queue");
    }
    profiling_ = enabled;
}

// Both timestamps are taken at the bottom of the pipe: the start once the
// commands before it have finished (the barrier between commands serializes
// them anyway), the end once this one has.
void Stream::begin_timestamp(VkCommandBuffer cmd, const std::string& label) {
    Slot& s = slot();
    timestamp_open_ = profiling_ && s.query_labels.size() < kQueryCapacity / 2;
    if (!timestamp_open_) return;
    const auto pair = static_cast<std::uint32_t>(s.query_labels.size());
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s.query_pool, 2 * pair);
    s.query_labels.push_back(label);
}

void Stream::end_timestamp(VkCommandBuffer cmd) {
    if (!timestamp_open_) return;
    Slot& s = slot();
    const auto pair = static_cast<std::uint32_t>(s.query_labels.size() - 1);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s.query_pool, 2 * pair + 1);
    timestamp_open_ = false;
}

void Stream::collect_timestamps(Slot& s) {
    const auto count = static_cast<std::uint32_t>(2 * s.query_labels.size());
    std::vector<std::uint64_t> ticks(count);
    check(vkGetQueryPoolResults(device_->device(), s.query_pool, 0, count,
                                ticks.size() * sizeof(std::uint64_t), ticks.data(),
                                sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT),
          "vkGetQueryPoolResults");
    const std::uint32_t bits = device_->timestamp_valid_bits();
    const std::uint64_t mask = bits >= 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << bits) - 1;
    const double period = device_->info().timestamp_period_ns;
    for (std::size_t i = 0; i < s.query_labels.size(); ++i) {
        const std::uint64_t elapsed = (ticks[2 * i + 1] - ticks[2 * i]) & mask;  // wraps are fine
        KernelStat& stat = profile_[s.query_labels[i]];
        ++stat.calls;
        stat.nanoseconds += static_cast<std::uint64_t>(double(elapsed) * period);
    }
    s.query_labels.clear();
    s.timestamps_pending = false;
}

VkDescriptorSet Stream::allocate_descriptor_set(VkDescriptorSetLayout layout) {
    const VkDevice vk = device_->device();
    Slot& s = slot();
    for (;;) {
        if (s.current_pool == s.descriptor_pools.size()) {
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
            s.descriptor_pools.push_back(pool);
        }

        const VkDescriptorSetAllocateInfo info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = s.descriptor_pools[s.current_pool],
            .descriptorSetCount = 1,
            .pSetLayouts = &layout};
        VkDescriptorSet set = VK_NULL_HANDLE;
        const VkResult result = vkAllocateDescriptorSets(vk, &info, &set);
        if (result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTED_POOL) {
            ++s.current_pool;
            continue;
        }
        check(result, "vkAllocateDescriptorSets");
        return set;
    }
}

void Stream::destroy() noexcept {
    const VkDevice vk = device_->device();
    for (Slot& s : slots_) {
        for (VkDescriptorPool pool : s.descriptor_pools) vkDestroyDescriptorPool(vk, pool, nullptr);
        s.descriptor_pools.clear();
        if (s.query_pool != VK_NULL_HANDLE) vkDestroyQueryPool(vk, s.query_pool, nullptr);
        if (s.command_pool != VK_NULL_HANDLE) vkDestroyCommandPool(vk, s.command_pool, nullptr);
        s.query_pool = VK_NULL_HANDLE;
        s.command_pool = VK_NULL_HANDLE;
        s.command_buffer = VK_NULL_HANDLE;
    }
    if (timeline_ != VK_NULL_HANDLE) vkDestroySemaphore(vk, timeline_, nullptr);
    timeline_ = VK_NULL_HANDLE;
}

}  // namespace vkml::hal
