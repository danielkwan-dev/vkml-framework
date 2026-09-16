#include <array>
#include <cstdint>
#include <string>

#include <vkml_shaders/matmul.spv.hpp>

#include "core/runtime.hpp"
#include "core/storage.hpp"
#include "vkml/ops.hpp"

namespace vkml {

namespace {

using detail::TensorAccess;

struct MatmulParams {
    std::uint32_t m, n, k;
};

std::uint32_t ceil_div(std::int64_t x, std::uint32_t d) {
    return static_cast<std::uint32_t>((x + d - 1) / d);
}

}  // namespace

Tensor matmul(const Tensor& a, const Tensor& b) {
    const Shape& as = a.shape();
    const Shape& bs = b.shape();
    if (as.size() < 2) {
        throw Error("matmul: a must have at least 2 dimensions, got " + to_string(as));
    }
    if (bs.size() != 2) throw Error("matmul: b must be a matrix, got " + to_string(bs));
    if (as.back() != bs.front()) {
        throw Error("matmul: cannot multiply " + to_string(as) + " by " + to_string(bs));
    }
    if (a.dtype() != DType::F32 || b.dtype() != DType::F32) {
        throw Error("matmul: no kernel for " + std::string(to_string(a.dtype())) + " x " +
                    std::string(to_string(b.dtype())) + " yet");
    }

    const std::int64_t k = as.back();
    const std::int64_t n = bs.back();
    std::int64_t rows = 1;  // leading dimensions of a fold into rows
    for (std::size_t i = 0; i + 1 < as.size(); ++i) rows *= as[i];

    Shape out_shape = as;
    out_shape.back() = n;
    detail::Runtime& runtime = TensorAccess::runtime(a);
    Tensor out = TensorAccess::empty(runtime, std::move(out_shape), DType::F32);
    if (out.numel() == 0) return out;
    if (k == 0) {  // every output is an empty sum, and a and b have no buffers to bind
        runtime.fill_zeros(TensorAccess::buffer(out));
        return out;
    }

    const std::uint32_t tile = runtime.matmul_tile();
    const std::array<std::uint32_t, 3> groups{ceil_div(n, tile), ceil_div(rows, tile), 1};
    const auto& max_groups = runtime.device.info().max_workgroup_count;
    if (groups[0] > max_groups[0] || groups[1] > max_groups[1]) {
        throw Error("matmul: output " + to_string(out.shape()) +
                    " needs more workgroups than the device allows");
    }

    const MatmulParams params{static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(n),
                              static_cast<std::uint32_t>(k)};
    const std::array<const hal::Buffer*, 3> buffers{
        &TensorAccess::buffer(a), &TensorAccess::buffer(b), &TensorAccess::buffer(out)};
    const hal::ComputePipeline& pipeline =
        runtime.pipeline(shaders::matmul, 3, sizeof(params), {tile});
    runtime.stream.dispatch(pipeline, buffers, std::as_bytes(std::span{&params, 1}), groups);
    return out;
}

}  // namespace vkml
