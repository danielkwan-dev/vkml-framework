#pragma once

// The one place vkml includes Vulkan. Nothing under include/ may include this.
//
// volk loads every entry point at runtime and defines VK_NO_PROTOTYPES, so it
// has to come before anything else that pulls in vulkan_core.h.
#include <volk.h>

// VMA fetches its function pointers through vkGet*ProcAddr, which volk provides.
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

#include <VkBootstrap.h>
