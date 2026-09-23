#pragma once

#include <cstdint>
#include <optional>

#include "vkml/tensor.hpp"

// Op variants with options the public API does not expose, for composite ops.
namespace vkml::detail {

// The TYPE_ constant of shaders/include/halves.glsl for a float dtype a kernel
// reads, or nullopt for dtypes kernels do not read.
inline std::optional<std::uint32_t> shader_type(DType dtype) {
    switch (dtype) {
        case DType::F32: return 0;
        case DType::F16: return 1;
        case DType::BF16: return 2;
        case DType::I32: break;
    }
    return std::nullopt;
}

// matmul or matmul_transposed. With b_group g > 1, a [B * g, M, K] and b of
// rank 3 with B batches: batch z of a is multiplied by batch z / g of b. With
// b_rows >= 0, a rank-3 b [B, R, C] is used as its first b_rows rows of each
// batch, [B, b_rows, C], still stepping batches by R * C elements.
Tensor matmul(const char* name, const Tensor& a, const Tensor& b, bool b_transposed,
              std::int64_t b_group = 1, std::int64_t b_rows = -1);

struct CausalMask {
    std::int64_t q_len;   // rows per head
    std::int64_t offset;  // position of query 0 among the columns (keys)
};

// softmax(scale * x) over the last dimension; with a mask, row r only sees
// columns <= (r % q_len) + offset.
Tensor softmax(const Tensor& x, float scale, const CausalMask* mask);

// Copies src [B, T, W] into rows start..start + T of every batch of dst
// [B, capacity, W], in place: appends to a KV cache.
void write_rows(const Tensor& dst, const Tensor& src, std::int64_t start);

// attention (see vkml/ops.hpp) with k and v read from the first kv_len rows of
// caches [kv_heads, capacity, head_dim].
Tensor attention(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache, std::int64_t kv_len,
                 bool causal);

}  // namespace vkml::detail
