#include <algorithm>
#include <array>
#include <cstdint>
#include <string>

#include <vkml_shaders/binary.spv.hpp>
#include <vkml_shaders/unary.spv.hpp>

#include "core/runtime.hpp"
#include "core/storage.hpp"
#include "vkml/ops.hpp"

namespace vkml {

namespace {

using detail::TensorAccess;

// Values of specialization constant 1 in shaders/binary.comp and unary.comp.
enum class BinaryOp : std::uint32_t { Add, Sub, Mul, Div };
enum class UnaryOp : std::uint32_t { Silu, Gelu };

struct ElementwiseParams {
    std::uint32_t count;
};

struct BinaryParams {
    std::uint32_t count;
    std::uint32_t b_count;  // b repeats every b_count elements
};

Tensor binary(BinaryOp op, const char* name, const Tensor& a, const Tensor& b) {
    // b may be a's trailing dimensions, repeated over the leading ones.
    const Shape& as = a.shape();
    const Shape& bs = b.shape();
    if (bs.size() > as.size() ||
        !std::equal(bs.begin(), bs.end(), as.end() - std::ptrdiff_t(bs.size()))) {
        throw Error(std::string(name) + ": shapes " + to_string(as) + " and " + to_string(bs) +
                    " differ, and the second is not the first's trailing dimensions");
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

    if (b.numel() == 0) throw Error(std::string(name) + ": the second operand is empty");
    const BinaryParams params{static_cast<std::uint32_t>(out.numel()),
                              static_cast<std::uint32_t>(b.numel())};
    const std::array<const hal::Buffer*, 3> buffers{
        &TensorAccess::buffer(a), &TensorAccess::buffer(b), &TensorAccess::buffer(out)};
    const hal::ComputePipeline& pipeline =
        runtime.pipeline("binary", shaders::binary, 3, sizeof(params),
                         {runtime.workgroup_width(), static_cast<std::uint32_t>(op)});
    runtime.stream.dispatch(pipeline, buffers, std::as_bytes(std::span{&params, 1}),
                            {runtime.workgroup_count(params.count), 1, 1});
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

    const ElementwiseParams params{static_cast<std::uint32_t>(out.numel())};
    const std::array<const hal::Buffer*, 2> buffers{&TensorAccess::buffer(x),
                                                    &TensorAccess::buffer(out)};
    const hal::ComputePipeline& pipeline =
        runtime.pipeline("unary", shaders::unary, 2, sizeof(params),
                         {runtime.workgroup_width(), static_cast<std::uint32_t>(op)});
    runtime.stream.dispatch(pipeline, buffers, std::as_bytes(std::span{&params, 1}),
                            {runtime.workgroup_count(params.count), 1, 1});
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
