#include "hal/pipeline.hpp"

#include <string>
#include <vector>

#include "hal/device.hpp"

namespace vkml::hal {

ComputePipeline::ComputePipeline(const Device& device, std::span<const std::uint32_t> spirv,
                                 std::uint32_t storage_buffer_count,
                                 std::uint32_t push_constant_bytes,
                                 std::span<const std::uint32_t> specialization)
    : device_(device.device()),
      storage_buffer_count_(storage_buffer_count),
      push_constant_bytes_(push_constant_bytes) {
    if (spirv.empty() || spirv.front() != kSpirvMagic) {
        throw Error("ComputePipeline: code is not SPIR-V (bad magic number)");
    }
    const std::uint32_t push_limit = device.info().max_push_constant_bytes;
    if (push_constant_bytes % 4 != 0 || push_constant_bytes > push_limit) {
        throw Error("ComputePipeline: push constants must be a multiple of 4 bytes and at most " +
                    std::to_string(push_limit) + ", got " + std::to_string(push_constant_bytes));
    }

    try {
        std::vector<VkDescriptorSetLayoutBinding> bindings(storage_buffer_count);
        for (std::uint32_t i = 0; i < storage_buffer_count; ++i) {
            bindings[i] = {.binding = i,
                           .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                           .descriptorCount = 1,
                           .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                           .pImmutableSamplers = nullptr};
        }
        const VkDescriptorSetLayoutCreateInfo set_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = storage_buffer_count,
            .pBindings = bindings.data()};
        check(vkCreateDescriptorSetLayout(device_, &set_info, nullptr, &set_layout_),
              "vkCreateDescriptorSetLayout");

        const VkPushConstantRange push_range{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = push_constant_bytes};
        const VkPipelineLayoutCreateInfo layout_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = 1,
            .pSetLayouts = &set_layout_,
            .pushConstantRangeCount = push_constant_bytes > 0 ? 1u : 0u,
            .pPushConstantRanges = &push_range};
        check(vkCreatePipelineLayout(device_, &layout_info, nullptr, &layout_),
              "vkCreatePipelineLayout");

        const VkShaderModuleCreateInfo module_info{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .codeSize = spirv.size_bytes(),
            .pCode = spirv.data()};
        VkShaderModule module = VK_NULL_HANDLE;
        check(vkCreateShaderModule(device_, &module_info, nullptr, &module), "vkCreateShaderModule");

        std::vector<VkSpecializationMapEntry> entries(specialization.size());
        for (std::uint32_t i = 0; i < entries.size(); ++i) {
            entries[i] = {.constantID = i,
                          .offset = i * static_cast<std::uint32_t>(sizeof(std::uint32_t)),
                          .size = sizeof(std::uint32_t)};
        }
        const VkSpecializationInfo spec_info{
            .mapEntryCount = static_cast<std::uint32_t>(entries.size()),
            .pMapEntries = entries.data(),
            .dataSize = specialization.size_bytes(),
            .pData = specialization.data()};

        const VkComputePipelineCreateInfo pipeline_info{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                      .pNext = nullptr,
                      .flags = 0,
                      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                      .module = module,
                      .pName = "main",
                      .pSpecializationInfo = specialization.empty() ? nullptr : &spec_info},
            .layout = layout_,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1};
        const VkResult result =
            vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline_);
        vkDestroyShaderModule(device_, module, nullptr);  // the pipeline keeps what it needs
        check(result, "vkCreateComputePipelines");
    } catch (...) {
        destroy();
        throw;
    }
}

ComputePipeline::~ComputePipeline() { destroy(); }

void ComputePipeline::destroy() noexcept {
    if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, pipeline_, nullptr);
    if (layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, layout_, nullptr);
    if (set_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr);
    pipeline_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE;
    set_layout_ = VK_NULL_HANDLE;
}

}  // namespace vkml::hal
