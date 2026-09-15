#include <array>
#include <cstdint>
#include <string>

#include "core/runtime.hpp"
#include "core/storage.hpp"
#include "vkml/ops.hpp"

namespace vkml {

namespace {

using detail::BinaryOp;
using detail::TensorAccess;
using detail::UnaryOp;

Tensor binary(BinaryOp op, const char* name, const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw Error(std::string(name) + ": shapes " + to_string(a.shape()) + " and " +
                    to_string(b.shape()) + " differ");
    }
    if (a.dtype() != b.dtype()) {
        throw Error(std::string(name) + ": dtypes " + std::string(to_string(a.dtype())) + " and " +
                    std::string(to_string(b.dtype())) + " differ");
    }
    if (a.dtype() != DType::F32) {
        throw Error(std::string(name) + ": no kernel for " + std::string(to_string(a.dtype())) +
                    " yet");
    }

    detail::Runtime& runtime = TensorAccess::runtime(a);
    Tensor out = TensorAccess::empty(runtime, a.shape(), a.dtype());
    if (out.numel() == 0) return out;

    const auto count = static_cast<std::uint32_t>(out.numel());
    const std::array<const hal::Buffer*, 3> buffers{
        &TensorAccess::buffer(a), &TensorAccess::buffer(b), &TensorAccess::buffer(out)};
    runtime.stream.dispatch(runtime.binary(op), buffers, std::as_bytes(std::span{&count, 1}),
                            {runtime.workgroup_count(count), 1, 1});
    return out;
}

Tensor unary(UnaryOp op, const char* name, const Tensor& x) {
    if (x.dtype() != DType::F32) {
        throw Error(std::string(name) + ": no kernel for " + std::string(to_string(x.dtype())) +
                    " yet");
    }

    detail::Runtime& runtime = TensorAccess::runtime(x);
    Tensor out = TensorAccess::empty(runtime, x.shape(), x.dtype());
    if (out.numel() == 0) return out;

    const auto count = static_cast<std::uint32_t>(out.numel());
    const std::array<const hal::Buffer*, 2> buffers{&TensorAccess::buffer(x),
                                                    &TensorAccess::buffer(out)};
    runtime.stream.dispatch(runtime.unary(op), buffers, std::as_bytes(std::span{&count, 1}),
                            {runtime.workgroup_count(count), 1, 1});
    return out;
}

}  // namespace

Tensor add(const Tensor& a, const Tensor& b) { return binary(BinaryOp::Add, "add", a, b); }
Tensor sub(const Tensor& a, const Tensor& b) { return binary(BinaryOp::Sub, "sub", a, b); }
Tensor mul(const Tensor& a, const Tensor& b) { return binary(BinaryOp::Mul, "mul", a, b); }
Tensor div(const Tensor& a, const Tensor& b) { return binary(BinaryOp::Div, "div", a, b); }

Tensor silu(const Tensor& x) { return unary(UnaryOp::Silu, "silu", x); }
Tensor gelu(const Tensor& x) { return unary(UnaryOp::Gelu, "gelu", x); }

}  // namespace vkml
