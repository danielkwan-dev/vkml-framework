# Third-party dependencies, pinned and fetched at configure time so that the
# only system requirements are a C++20 compiler, CMake, Ninja and glslc.
#
# SYSTEM marks their headers as system headers, so our warning flags (and
# -Werror in CI) apply to our code only.

include(FetchContent)

FetchContent_Declare(VulkanHeaders
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG        v1.4.363
    GIT_SHALLOW    TRUE
    SYSTEM)

# vk-bootstrap: instance creation and physical-device selection.
# Its version tracks Vulkan-Headers; keep the two tags in step.
FetchContent_Declare(vk_bootstrap
    GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git
    GIT_TAG        v1.4.363
    GIT_SHALLOW    TRUE
    SYSTEM)

# volk: loads Vulkan entry points at runtime, so nothing links against the
# loader and a machine without Vulkan fails with a clear error, not at startup.
FetchContent_Declare(volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG        vulkan-sdk-1.4.357.0
    GIT_SHALLOW    TRUE
    SYSTEM)

# VMA: device memory allocation and suballocation.
FetchContent_Declare(VulkanMemoryAllocator
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG        v3.4.0
    GIT_SHALLOW    TRUE
    SYSTEM)

# nlohmann/json: parses the JSON header of safetensors weight files.
FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.12.0
    GIT_SHALLOW    TRUE
    SYSTEM)
set(JSON_Install OFF)

# volk would otherwise go looking for a Vulkan SDK; we hand it the headers below.
set(VOLK_PULL_IN_VULKAN OFF)

FetchContent_MakeAvailable(VulkanHeaders vk_bootstrap volk VulkanMemoryAllocator nlohmann_json)

target_link_libraries(volk PUBLIC Vulkan::Headers)

# Everything the HAL needs, in one target, for vkml and for tests that reach
# into src/hal directly.
add_library(vkml_hal_deps INTERFACE)
target_link_libraries(vkml_hal_deps INTERFACE
    Vulkan::Headers
    volk::volk
    vk-bootstrap::vk-bootstrap
    GPUOpen::VulkanMemoryAllocator)
