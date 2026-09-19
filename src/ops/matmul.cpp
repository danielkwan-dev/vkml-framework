#include <algorithm>
#include <array>
#include <cstdint>
#include <string>

#include <vkml_shaders/matmul.spv.hpp>

#include "core/runtime.hpp"
#include "core/storage.hpp"
#include "ops/internal.hpp"
#include "vkml/ops.hpp"

namespace vkml {

namespace {

using detail::TensorAccess;

struct MatmulParams {
    std::uint32_t m, n, k;
    std::uint32_t b_stride;
    std::uint32_t b_group;
};

std::uint32_t ceil_div(std::int64_t x, std::uint32_t d) {
    return static_cast<std::uint32_t>((x + d - 1) / d);
}

std::int64_t product(const Shape& shape, std::size_t count) {
    std::int64_t p = 1;
    for (std::size_t i = 0; i < count; ++i) p *= shape[i];
    return p;
}

}  // namespace

// a [..., M, K] times b [K, N], or [N, K] when b_transposed. A 2-D b is shared
// by every row of a; a higher-rank b must have a's batch dimensions exactly,
// or with b_group > 1 be rank 3 with 1 / b_group of a's batches.
Tensor detail::matmul(const char* name, const Tensor& a, const Tensor& b, bool b_transposed,
                      std::int64_t b_group, std::int64_t b_rows) {
    const Shape& as = a.shape();
    Shape bs = b.shape();  // the logical shape, which b_rows may shorten
    const std::string b_desc = (b_transposed ? "transposed " : "") + to_string(bs);
    if (b_rows >= 0) {
        if (bs.size() != 3 || b_rows > bs[1]) {
            throw Error(std::string(name) + ": cannot use the first " + std::to_string(b_rows) +
                        " rows of " + b_desc);
        }
        bs[1] = b_rows;
    }
    if (as.size() < 2 || bs.size() < 2) {
        throw Error(std::string(name) + ": both operands need at least 2 dimensions, got " +
                    to_string(as) + " and " + to_string(bs));
    }
    if (a.dtype() != DType::F32 || b.dtype() != DType::F32) {
        throw Error(std::string(name) + ": no kernel for " + std::string(to_string(a.dtype())) +
                    " x " + std::string(to_string(b.dtype())) + " yet");
    }

    const std::size_t b_rank = bs.size();
    // Elements between batches of b in memory, whichever rows are used.
    const std::int64_t b_batch_elements = b.shape()[b_rank - 2] * b.shape()[b_rank - 1];
    const std::int64_t k = as.back();
    const std::int64_t b_k = b_transposed ? bs[b_rank - 1] : bs[b_rank - 2];
    const std::int64_t n = b_transposed ? bs[b_rank - 2] : bs[b_rank - 1];
    if (k != b_k) {
        throw Error(std::string(name) + ": cannot multiply " + to_string(as) + " by " + b_desc);
    }

    std::int64_t batches = 1;
    std::int64_t m = product(as, as.size() - 1);  // a shared b: leading dims fold into rows
    std::int64_t b_stride = 0;
    if (b_group > 1) {
        if (as.size() != 3 || b_rank != 3 || as[0] != bs[0] * b_group) {
            throw Error(std::string(name) + ": " + to_string(as) + " does not have " +
                        std::to_string(b_group) + " batches per batch of " + b_desc);
        }
        batches = as[0];
        m = as[1];
        b_stride = b_batch_elements;
    } else if (b_rank > 2) {
        if (as.size() != b_rank || !std::equal(as.begin(), as.end() - 2, bs.begin())) {
            throw Error(std::string(name) + ": batch dimensions of " + to_string(as) + " and " +
                        b_desc + " differ");
        }
        batches = product(as, as.size() - 2);
        m = as[as.size() - 2];
        b_stride = b_batch_elements;
    }

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
    const std::array<std::uint32_t, 3> groups{ceil_div(n, tile), ceil_div(m, tile),
                                              static_cast<std::uint32_t>(batches)};
    const auto& max_groups = runtime.device.info().max_workgroup_count;
    for (std::size_t i = 0; i < 3; ++i) {
        if (groups[i] > max_groups[i]) {
            throw Error(std::string(name) + ": output " + to_string(out.shape()) +
                        " needs more workgroups than the device allows");
        }
    }

    const MatmulParams params{static_cast<std::uint32_t>(m), static_cast<std::uint32_t>(n),
                              static_cast<std::uint32_t>(k), static_cast<std::uint32_t>(b_stride),
                              static_cast<std::uint32_t>(b_group)};
    const std::array<const hal::Buffer*, 3> buffers{
        &TensorAccess::buffer(a), &TensorAccess::buffer(b), &TensorAccess::buffer(out)};
    const hal::ComputePipeline& pipeline = runtime.pipeline(
        shaders::matmul, 3, sizeof(params), {tile, static_cast<std::uint32_t>(b_transposed)});
    runtime.stream.dispatch(pipeline, buffers, std::as_bytes(std::span{&params, 1}), groups);
    return out;
}

Tensor matmul(const Tensor& a, const Tensor& b) { return detail::matmul("matmul", a, b, false); }

Tensor matmul_transposed(const Tensor& a, const Tensor& b) {
    return detail::matmul("matmul_transposed", a, b, true);
}

}  // namespace vkml
