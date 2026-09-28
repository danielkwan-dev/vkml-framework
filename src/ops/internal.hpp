#pragma once

#include <cstdint>
#include <optional>
#include <utility>

#include "vkml/ops.hpp"
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

// The TYPE_ constant of shaders/include/halves.glsl for a quantized format.
inline std::uint32_t quant_shader_type(QuantType type) { return type == QuantType::q4_0 ? 4 : 3; }

// How many values of a quantized format one i32 element packs.
inline std::int64_t quant_values_per_word(QuantType type) {
    return type == QuantType::q4_0 ? 8 : 4;
}

// matmul or matmul_transposed. With b_group g > 1, a [B * g, M, K] and b of
// rank 3 with B batches: batch z of a is multiplied by batch z / g of b. With
// b_rows >= 0, a rank-3 b [B, R, C] is used as its first b_rows rows of each
// batch, [B, b_rows, C], still stepping batches by R * C elements.
Tensor matmul(const char* name, const Tensor& a, const Tensor& b, bool b_transposed,
              std::int64_t b_group = 1, std::int64_t b_rows = -1);

// An input a to several matmul_transposed calls, such as the q, k and v
// projections of one layer: with quantized weights and integer dot products,
// a is quantized once, on first use, instead of once per call.
class SharedInput {
public:
    explicit SharedInput(Tensor a) : a_(std::move(a)) {}

    // matmul_transposed(a, b).
    Tensor times_transposed(const Tensor& b) const;
    Tensor times_transposed(const QuantizedMatrix& b);

private:
    Tensor a_;
    std::optional<QuantizedMatrix> a_q8_;
};

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

// Rows start..start + count of every batch of src [B, R, W], as [B, count, W].
Tensor read_rows(const Tensor& src, std::int64_t start, std::int64_t count);

// Scores matrices beyond this many bytes are computed a chunk of queries at a
// time (see attention).
inline constexpr std::int64_t kMaxScoreBytes = std::int64_t{64} << 20;

// attention (see vkml/ops.hpp) with k and v read from the first kv_len rows of
// caches [kv_heads, capacity, head_dim]. Many queries whose scores would take
// more than max_score_bytes are taken in chunks, each against only the keys
// it can see when causal.
Tensor attention(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache, std::int64_t kv_len,
                 bool causal, std::int64_t max_score_bytes = kMaxScoreBytes);

}  // namespace vkml::detail
