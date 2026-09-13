#pragma once

#include <optional>
#include <utility>

#include "core/runtime.hpp"
#include "hal/buffer.hpp"
#include "vkml/tensor.hpp"

namespace vkml::detail {

// The GPU allocation behind one or more Tensors. Empty tensors have no buffer,
// since Vulkan has no zero-sized buffers. Otherwise the buffer is rounded up
// to whole 32-bit words so word-wise kernels such as fill never go past it.
struct Storage {
    Runtime* runtime;
    std::optional<hal::Buffer> buffer;

    // Pending GPU work may still use the buffer, so it is retired, not freed.
    ~Storage() {
        if (!buffer) return;
        try {
            runtime->retire(std::move(*buffer));
        } catch (...) {
            // Only a lost device or exhausted host memory gets here; freeing
            // at once is then the least bad option.
        }
    }
};

// What op implementations need from a Tensor beyond its public interface.
struct TensorAccess {
    static Runtime& runtime(const Tensor& t) { return *t.storage_->runtime; }
    static const hal::Buffer& buffer(const Tensor& t) { return *t.storage_->buffer; }
    static Tensor empty(Runtime& runtime, Shape shape, DType dtype) {
        return Tensor::empty(runtime, std::move(shape), dtype);
    }
};

}  // namespace vkml::detail
