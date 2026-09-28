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
// until submit(), which returns a timeline value that wait() blocks on.
//
// Every command is ordered after the one before it by a full memory barrier,
// also across submissions, so later work always sees earlier results. Work
// completed by wait() is visible to the host after Buffer::invalidate().
//
// Submissions rotate through a few command buffers, each with its own
// descriptor pools, so the host can record the next submission while earlier
// ones run: submitting part of a long sequence early lets the GPU start on it.
// A command buffer is reused once the submission holding it has finished.
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

    std::uint64_t submitted() const noexcept { return submitted_; }  // the last submission's value

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
    // What one submission records into; reused once that submission finishes.
    struct Slot {
        VkCommandPool command_pool = VK_NULL_HANDLE;
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        std::vector<VkDescriptorPool> descriptor_pools;
        std::size_t current_pool = 0;
        std::uint64_t value = 0;  // the submission that last used it, 0 before any
        VkQueryPool query_pool = VK_NULL_HANDLE;
        std::vector<std::string> query_labels;  // one per start/end pair recorded
        bool timestamps_pending = false;        // submitted, timestamps not yet read
    };
    static constexpr std::size_t kSlots = 3;

    Slot& slot() noexcept { return slots_[current_]; }
    VkCommandBuffer begin_command();
    void begin_timestamp(VkCommandBuffer cmd, const std::string& label);
    void end_timestamp(VkCommandBuffer cmd);
    void collect_timestamps(Slot& s);
    VkDescriptorSet allocate_descriptor_set(VkDescriptorSetLayout layout);
    void destroy() noexcept;

    const Device* device_;
    std::array<Slot, kSlots> slots_;
    std::size_t current_ = 0;
    VkSemaphore timeline_ = VK_NULL_HANDLE;
    std::uint64_t submitted_ = 0;
    bool recording_ = false;
    bool barrier_needed_ = false;  // earlier commands exist that the next must follow

    bool profiling_ = false;
    bool timestamp_open_ = false;  // a start written, its end not yet
    std::map<std::string, KernelStat> profile_;
};

}  // namespace vkml::hal
