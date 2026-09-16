#include "vkml/tensor.hpp"

#include <algorithm>
#include <limits>
#include <utility>

#include "core/runtime.hpp"
#include "core/storage.hpp"
#include "hal/buffer.hpp"

namespace vkml {

namespace {

std::int64_t checked_numel(const Shape& shape) {
    std::int64_t n = 1;
    for (const std::int64_t dim : shape) {
        if (dim < 0) {
            throw Error("Tensor: shape " + to_string(shape) + " has a negative dimension");
        }
        if (dim != 0 && n > std::numeric_limits<std::int64_t>::max() / dim) {
            throw Error("Tensor: shape " + to_string(shape) + " has too many elements");
        }
        n *= dim;
    }
    return n;
}

constexpr std::uint64_t round_up_to_word(std::uint64_t bytes) { return (bytes + 3) / 4 * 4; }

}  // namespace

std::string to_string(const Shape& shape) {
    std::string out = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) out += ", ";
        out += std::to_string(shape[i]);
    }
    return out + "]";
}

std::string_view to_string(DType dtype) noexcept {
    switch (dtype) {
        case DType::F32: return "f32";
        case DType::F16: return "f16";
        case DType::I32: break;
    }
    return "i32";
}

Tensor::Tensor(std::shared_ptr<detail::Storage> storage, Shape shape, DType dtype,
               std::int64_t numel)
    : storage_(std::move(storage)), shape_(std::move(shape)), dtype_(dtype), numel_(numel) {}

Tensor Tensor::empty(Context& context, Shape shape, DType dtype) {
    return empty(context.runtime(), std::move(shape), dtype);
}

Tensor Tensor::empty(detail::Runtime& runtime, Shape shape, DType dtype) {
    const std::int64_t numel = checked_numel(shape);
    auto storage = std::make_shared<detail::Storage>(&runtime);
    if (numel > 0) {
        const auto count = static_cast<std::uint64_t>(numel);
        const std::uint64_t limit = runtime.device.info().max_storage_buffer_range;
        if (count > limit / element_size(dtype)) {
            throw Error("Tensor: shape " + to_string(shape) + " of " +
                        std::string(to_string(dtype)) + " is larger than maxStorageBufferRange (" +
                        std::to_string(limit) + " bytes)");
        }
        storage->buffer.emplace(runtime.device, round_up_to_word(count * element_size(dtype)),
                                hal::MemoryUsage::DeviceLocal);
    }
    return Tensor{std::move(storage), std::move(shape), dtype, numel};
}

Tensor Tensor::zeros(Context& context, Shape shape, DType dtype) {
    Tensor t = empty(context, std::move(shape), dtype);
    if (t.numel_ > 0) context.runtime().fill_zeros(*t.storage_->buffer);
    return t;
}

Tensor Tensor::from_bytes(Context& context, std::span<const std::byte> bytes, Shape shape,
                          DType dtype) {
    const std::int64_t numel = checked_numel(shape);
    const std::uint64_t expected = static_cast<std::uint64_t>(numel) * element_size(dtype);
    if (bytes.size() != expected) {
        throw Error("Tensor: shape " + to_string(shape) + " of " + std::string(to_string(dtype)) +
                    " needs " + std::to_string(numel) + " elements (" + std::to_string(expected) +
                    " bytes), got " + std::to_string(bytes.size()) + " bytes");
    }

    Tensor t = empty(context, std::move(shape), dtype);
    if (numel == 0) return t;

    detail::Runtime& runtime = context.runtime();
    hal::Buffer staging{runtime.device, bytes.size(), hal::MemoryUsage::Upload};
    std::memcpy(staging.mapped(), bytes.data(), bytes.size());
    staging.flush();
    runtime.stream.copy(staging, *t.storage_->buffer, bytes.size());
    runtime.retire(std::move(staging));
    return t;
}

Tensor Tensor::reshape(Shape shape) const {
    const auto inferred = std::ranges::find(shape, -1);
    if (inferred != shape.end()) {
        if (std::ranges::count(shape, -1) > 1) {
            throw Error("Tensor::reshape: " + to_string(shape) + " has more than one -1");
        }
        *inferred = 1;
        const std::int64_t rest = checked_numel(shape);
        *inferred = -1;  // shown as written in the messages below
        if (rest == 0) {
            throw Error("Tensor::reshape: the -1 in " + to_string(shape) +
                        " is ambiguous next to a 0");
        }
        if (numel_ % rest != 0) {
            throw Error("Tensor::reshape: cannot view " + to_string(shape_) + " as " +
                        to_string(shape));
        }
        *inferred = numel_ / rest;
    }
    if (checked_numel(shape) != numel_) {
        throw Error("Tensor::reshape: cannot view " + to_string(shape_) + " as " +
                    to_string(shape));
    }
    return Tensor{storage_, std::move(shape), dtype_, numel_};
}

std::vector<std::byte> Tensor::to_bytes() const {
    const std::uint64_t size = nbytes();
    if (size == 0) return {};

    detail::Runtime& runtime = *storage_->runtime;
    hal::Buffer staging{runtime.device, size, hal::MemoryUsage::Readback};
    runtime.stream.copy(*storage_->buffer, staging, size);
    runtime.synchronize();
    staging.invalidate();

    std::vector<std::byte> out(size);
    std::memcpy(out.data(), staging.mapped(), size);
    return out;
}

}  // namespace vkml
