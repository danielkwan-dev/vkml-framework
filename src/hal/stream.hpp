#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <vector>

#include "hal/vulkan.hpp"

namespace vkml::hal {

class Buffer;
class ComputePipeline;
class Device;

// An in-order sequence of GPU work on the compute queue. Commands are recorded
// into one command buffer until submit(), which returns a timeline value that
// wait() blocks on.
//
// Every command is ordered after the one before it by a full memory barrier,
// so later work always sees earlier results; finer-grained barriers are a
// later optimization. Work completed by wait() is visible to the host after
// Buffer::invalidate().
//
// Recording after a submit waits for that submission to finish before reusing
// the command buffer and descriptor sets, so host and GPU do not yet overlap.
class Stream {
public:
    explicit Stream(const Device& device);
    ~Stream();  // waits for submitted work; unsubmitted commands are discarded

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    // Binds buffers[i] to binding i and launches groups workgroups. Buffers
    // must outlive the submission. Throws before recording anything if the
    // arguments do not match the pipeline or the device limits.
    void dispatch(const ComputePipeline& pipeline, std::span<const Buffer* const> buffers,
                  std::span<const std::byte> push_constants, std::array<std::uint32_t, 3> groups);

    // Copies the first bytes of src to the start of dst.
    void copy(const Buffer& src, const Buffer& dst, std::uint64_t bytes);

    // Submits everything recorded so far. With nothing recorded, returns the
    // value of the previous submission (0 before the first).
    std::uint64_t submit();

    void wait(std::uint64_t value);
    std::uint64_t completed() const;

    // The value that covers everything recorded so far: the next submit()'s
    // while commands are pending, else the last one's.
    std::uint64_t pending_value() const noexcept {
        return recording_ ? submitted_ + 1 : submitted_;
    }

    void synchronize() { wait(submit()); }

    // Profiling: with it on, each dispatch and copy is bracketed by GPU
    // timestamps, totalled per pipeline label ("copy" for copies) once the
    // submission holding them has been waited for. Turning it on or off
    // finishes the work already recorded first.
    struct KernelStat {
        std::uint64_t calls = 0;
        std::uint64_t nanoseconds = 0;
    };
    void set_profiling(bool enabled);
    const std::map<std::string, KernelStat>& profile() const noexcept { return profile_; }
    void reset_profile() { profile_.clear(); }

private:
    VkCommandBuffer begin_command();
    void begin_timestamp(VkCommandBuffer cmd, const std::string& label);
    void end_timestamp(VkCommandBuffer cmd);
    void collect_timestamps();
    VkDescriptorSet allocate_descriptor_set(VkDescriptorSetLayout layout);
    void destroy() noexcept;

    const Device* device_;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorPool> descriptor_pools_;
    std::size_t current_pool_ = 0;
    VkSemaphore timeline_ = VK_NULL_HANDLE;
    std::uint64_t submitted_ = 0;
    bool recording_ = false;
    bool empty_ = true;  // nothing recorded since begin, so no barrier needed yet

    bool profiling_ = false;
    VkQueryPool query_pool_ = VK_NULL_HANDLE;
    std::vector<std::string> query_labels_;  // one per start/end pair in the recording
    bool timestamp_open_ = false;            // a start written, its end not yet
    std::uint64_t timestamps_value_ = 0;     // the submission whose timestamps await reading
    std::map<std::string, KernelStat> profile_;
};

}  // namespace vkml::hal
