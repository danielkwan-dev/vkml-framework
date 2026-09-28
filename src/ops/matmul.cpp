#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <string>

#include <vkml_shaders/gemv.spv.hpp>
#include <vkml_shaders/matmul.spv.hpp>

#include "core/runtime.hpp"
#include "core/storage.hpp"
#include "ops/internal.hpp"
#include "vkml/ops.hpp"

namespace vkml {

namespace {

using detail::TensorAccess;

// Matrix-vector shapes (decoding) use shaders/gemv.comp, which reads b once at
// close to memory bandwidth for up to this many rows of a; more rows use the
// tiled kernel.
// Measured on Iris Xe: at 8 rows it still beats the tiled kernel by 30%.
constexpr std::int64_t kGemvMaxRows = 8;           // the largest MAX_M in shaders/gemv.comp
constexpr std::uint32_t kGemvOutputsPerGroup = 4;  // ROWS in shaders/gemv.comp

// The block shape of shaders/matmul.comp: each invocation computes tm rows and
// tn columns of c, stepping through k by bk, so a workgroup of tile x tile
// covers tile * tm rows and tile * tn columns. The shader takes these as
// specialization constants and the dispatch is sized from the same values.
struct MatmulBlock {
    std::uint32_t tm, tn, bk;
};

// Taller blocks reuse more of each loaded value but waste work on short a.
// Chosen from a sweep on Iris Xe (vkml-bench, results checked): 1 x 4 was
// fastest for 8 rows, 2 x 4 for 32, and 4 x 4 with bk 8 for 128, at about 390
// GFLOP/s, three times the one-output-per-invocation kernel it replaced.
MatmulBlock matmul_block(std::int64_t rows) {
    if (rows < 32) return {1, 4, 16};
    if (rows < 64) return {2, 4, 16};
    return {4, 4, 8};
}

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

// What a matmul launch needs besides its buffers.
struct Launch {
    std::uint32_t b_type;
    bool b_transposed;
    std::int64_t m, n, k;
    std::int64_t batches = 1;
    std::int64_t b_stride = 0;
    std::int64_t b_group = 1;
};

// Records gemv or the tiled kernel writing out. scales holds Q8_0 block scales;
// for other weight types any buffer will do, as the kernels do not read it.
void launch(const char* name, const Tensor& out, const hal::Buffer& a, const hal::Buffer& b,
            const hal::Buffer& scales, const Launch& l) {
    detail::Runtime& runtime = TensorAccess::runtime(out);
    const bool gemv = l.b_transposed && l.m <= kGemvMaxRows;
    const MatmulBlock block = matmul_block(l.m);
    const std::uint32_t tile = runtime.matmul_tile();
    const std::array<std::uint32_t, 3> groups =
        // Each gemv workgroup computes its outputs for every row of a.
        gemv ? std::array{ceil_div(l.n, kGemvOutputsPerGroup), 1u,
                          static_cast<std::uint32_t>(l.batches)}
             : std::array{ceil_div(l.n, tile * block.tn), ceil_div(l.m, tile * block.tm),
                          static_cast<std::uint32_t>(l.batches)};
    const auto& max_groups = runtime.device.info().max_workgroup_count;
    for (std::size_t i = 0; i < 3; ++i) {
        if (groups[i] > max_groups[i]) {
            throw Error(std::string(name) + ": output " + to_string(out.shape()) +
                        " needs more workgroups than the device allows");
        }
    }

    const MatmulParams params{static_cast<std::uint32_t>(l.m), static_cast<std::uint32_t>(l.n),
                              static_cast<std::uint32_t>(l.k),
                              static_cast<std::uint32_t>(l.b_stride),
                              static_cast<std::uint32_t>(l.b_group)};
    const std::array<const hal::Buffer*, 4> buffers{&a, &b, &TensorAccess::buffer(out), &scales};
    const hal::ComputePipeline& pipeline =
        gemv ? runtime.pipeline("gemv", shaders::gemv, 4, sizeof(params),
                                {kGemvOutputsPerGroup, l.b_type, std::bit_ceil(std::uint32_t(l.m))})
             : runtime.pipeline("matmul", shaders::matmul, 4, sizeof(params),
                                {tile, static_cast<std::uint32_t>(l.b_transposed), l.b_type,
                                 block.tm, block.tn, block.bk});
    runtime.stream.dispatch(pipeline, buffers, std::as_bytes(std::span{&params, 1}), groups);
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
    const auto b_type = shader_type(b.dtype());  // 16-bit weights are read as they are
    if (a.dtype() != DType::F32 || !b_type) {
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

    Launch l{.b_type = *b_type,
             .b_transposed = b_transposed,
             .m = product(as, as.size() - 1),  // a shared b: leading dims fold into rows
             .n = n,
             .k = k};
    if (b_group > 1) {
        if (as.size() != 3 || b_rank != 3 || as[0] != bs[0] * b_group) {
            throw Error(std::string(name) + ": " + to_string(as) + " does not have " +
                        std::to_string(b_group) + " batches per batch of " + b_desc);
        }
        l.batches = as[0];
        l.m = as[1];
        l.b_stride = b_batch_elements;
        l.b_group = b_group;
    } else if (b_rank > 2) {
        if (as.size() != b_rank || !std::equal(as.begin(), as.end() - 2, bs.begin())) {
            throw Error(std::string(name) + ": batch dimensions of " + to_string(as) + " and " +
                        b_desc + " differ");
        }
        l.batches = product(as, as.size() - 2);
        l.m = as[as.size() - 2];
        l.b_stride = b_batch_elements;
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
    const hal::Buffer& b_buffer = TensorAccess::buffer(b);
    launch(name, out, TensorAccess::buffer(a), b_buffer, b_buffer, l);
    return out;
}

Tensor matmul(const Tensor& a, const Tensor& b) { return detail::matmul("matmul", a, b, false); }

Tensor matmul_transposed(const Tensor& a, const Tensor& b) {
    return detail::matmul("matmul_transposed", a, b, true);
}

Tensor matmul_transposed(const Tensor& a, const QuantizedMatrix& b) {
    const Shape& as = a.shape();
    if (as.size() < 2 || a.dtype() != DType::F32 || as.back() != b.cols) {
        throw Error("matmul_transposed: cannot multiply " + std::string(to_string(a.dtype())) +
                    " " + to_string(as) + " by a quantized matrix of " + std::to_string(b.rows) +
                    " rows and " + std::to_string(b.cols) + " columns");
    }
    Shape out_shape = as;
    out_shape.back() = b.rows;
    Tensor out = TensorAccess::empty(TensorAccess::runtime(a), std::move(out_shape), DType::F32);
    if (out.numel() == 0) return out;
    launch("matmul_transposed", out, TensorAccess::buffer(a), TensorAccess::buffer(b.values),
           TensorAccess::buffer(b.scales),
           Launch{.b_type = detail::quant_shader_type(b.type),
                  .b_transposed = true,
                  .m = product(as, as.size() - 1),
                  .n = b.rows,
                  .k = b.cols});
    return out;
}

}  // namespace vkml
