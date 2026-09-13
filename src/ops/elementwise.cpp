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

}  // namespace

Tensor add(const Tensor& a, const Tensor& b) { return binary(BinaryOp::Add, "add", a, b); }
Tensor sub(const Tensor& a, const Tensor& b) { return binary(BinaryOp::Sub, "sub", a, b); }
Tensor mul(const Tensor& a, const Tensor& b) { return binary(BinaryOp::Mul, "mul", a, b); }
Tensor div(const Tensor& a, const Tensor& b) { return binary(BinaryOp::Div, "div", a, b); }

}  // namespace vkml
