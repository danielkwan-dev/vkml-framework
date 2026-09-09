#pragma once

#include <stdexcept>

namespace vkml {

// Thrown for every failure vkml reports: no suitable GPU, a failed Vulkan call,
// invalid arguments. The message says what was being attempted.
class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

}  // namespace vkml
