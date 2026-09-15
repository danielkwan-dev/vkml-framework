#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "vkml/context.hpp"
#include "vkml/dtype.hpp"
#include "vkml/error.hpp"

namespace vkml {

// Dimension sizes, outermost first. {} is a scalar; any 0 makes an empty tensor.
using Shape = std::vector<std::int64_t>;

std::string to_string(const Shape& shape);  // "[2, 3]"

namespace detail {

class Runtime;
struct Storage;
struct TensorAccess;

template <class T>
struct DTypeOf;
template <>
struct DTypeOf<float> {
    static constexpr DType value = DType::F32;
};
template <>
struct DTypeOf<std::int32_t> {
    static constexpr DType value = DType::I32;
};

}  // namespace detail

// A dense, row-major array in GPU memory.
//
// Copying a Tensor is cheap and shares the storage, as in PyTorch. A tensor
// must not outlive the Context that made it.
//
// Operations are recorded for the GPU and return at once; only reading a
// tensor back (to_bytes, to_vector) waits for the work it depends on.
class Tensor {
public:
    // Uninitialized contents.
    static Tensor empty(Context& context, Shape shape, DType dtype);
    static Tensor zeros(Context& context, Shape shape, DType dtype);

    // bytes must hold exactly the shape's elements of dtype, in row-major order.
    static Tensor from_bytes(Context& context, std::span<const std::byte> bytes, Shape shape,
                             DType dtype);

    template <class T>
    static Tensor from_data(Context& context, std::span<const T> data, Shape shape) {
        return from_bytes(context, std::as_bytes(data), std::move(shape), detail::DTypeOf<T>::value);
    }

    std::vector<std::byte> to_bytes() const;

    // Throws unless T matches dtype().
    template <class T>
    std::vector<T> to_vector() const {
        if (dtype_ != detail::DTypeOf<T>::value) {
            throw Error("Tensor::to_vector: tensor holds " + std::string(to_string(dtype_)) +
                        ", not " + std::string(to_string(detail::DTypeOf<T>::value)));
        }
        const std::vector<std::byte> bytes = to_bytes();
        std::vector<T> out(static_cast<std::size_t>(numel_));
        if (!bytes.empty()) std::memcpy(out.data(), bytes.data(), bytes.size());
        return out;
    }

    const Shape& shape() const noexcept { return shape_; }
    DType dtype() const noexcept { return dtype_; }
    std::int64_t numel() const noexcept { return numel_; }
    std::uint64_t nbytes() const noexcept {
        return static_cast<std::uint64_t>(numel_) * element_size(dtype_);
    }

    // The same storage under another shape with the same element count. One
    // dimension may be -1, meaning whatever size makes the counts match.
    Tensor reshape(Shape shape) const;

    bool shares_storage_with(const Tensor& other) const noexcept {
        return storage_ == other.storage_;
    }

private:
    friend struct detail::TensorAccess;

    Tensor(std::shared_ptr<detail::Storage> storage, Shape shape, DType dtype, std::int64_t numel);
    static Tensor empty(detail::Runtime& runtime, Shape shape, DType dtype);

    std::shared_ptr<detail::Storage> storage_;
    Shape shape_;
    DType dtype_;
    std::int64_t numel_;
};

}  // namespace vkml
