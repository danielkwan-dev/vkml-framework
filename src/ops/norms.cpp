#include <algorithm>
#include <array>
#include <cstdint>
#include <string>

#include "core/runtime.hpp"
#include "core/storage.hpp"
#include "vkml/ops.hpp"

namespace vkml {

namespace {

using detail::TensorAccess;

struct RowShape {
    std::uint32_t rows;
    std::uint32_t cols;
};

// Folds every dimension but the last into rows.
RowShape row_shape(const char* name, const Tensor& x) {
    if (x.shape().empty()) {
        throw Error(std::string(name) + ": input must have at least 1 dimension, got a scalar");
    }
    if (x.dtype() != DType::F32) {
        throw Error(std::string(name) + ": no kernel for " + std::string(to_string(x.dtype())) +
                    " yet");
    }
    const std::int64_t cols = x.shape().back();
    return {static_cast<std::uint32_t>(cols == 0 ? 0 : x.numel() / cols),
            static_cast<std::uint32_t>(cols)};
}

// One workgroup per row, up to the device limit; the kernels stride over the rest.
std::array<std::uint32_t, 3> row_groups(const detail::Runtime& runtime, std::uint32_t rows) {
    return {std::min(rows, runtime.device.info().max_workgroup_count[0]), 1, 1};
}

}  // namespace

Tensor softmax(const Tensor& x) {
    const RowShape rs = row_shape("softmax", x);
    detail::Runtime& runtime = TensorAccess::runtime(x);
    Tensor out = TensorAccess::empty(runtime, x.shape(), DType::F32);
    if (out.numel() == 0) return out;

    const std::array<const hal::Buffer*, 2> buffers{&TensorAccess::buffer(x),
                                                    &TensorAccess::buffer(out)};
    runtime.stream.dispatch(runtime.softmax(), buffers, std::as_bytes(std::span{&rs, 1}),
                            row_groups(runtime, rs.rows));
    return out;
}

Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps) {
    const RowShape rs = row_shape("rms_norm", x);
    if (weight.shape() != Shape{x.shape().back()} || weight.dtype() != DType::F32) {
        throw Error("rms_norm: weight must be f32 " + to_string(Shape{x.shape().back()}) +
                    " to match x " + to_string(x.shape()) + ", got " +
                    std::string(to_string(weight.dtype())) + " " + to_string(weight.shape()));
    }
    detail::Runtime& runtime = TensorAccess::runtime(x);
    Tensor out = TensorAccess::empty(runtime, x.shape(), DType::F32);
    if (out.numel() == 0) return out;

    struct {
        RowShape shape;
        float eps;
    } const params{rs, eps};
    static_assert(sizeof(params) == 12);
    const std::array<const hal::Buffer*, 3> buffers{
        &TensorAccess::buffer(x), &TensorAccess::buffer(weight), &TensorAccess::buffer(out)};
    runtime.stream.dispatch(runtime.rms_norm(), buffers, std::as_bytes(std::span{&params, 1}),
                            row_groups(runtime, rs.rows));
    return out;
}

}  // namespace vkml
