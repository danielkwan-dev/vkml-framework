#pragma once

#include <cstdint>
#include <vector>

#include "vkml/context.hpp"
#include "vkml/tensor.hpp"

namespace vkml {

// Elementwise arithmetic on two tensors of the same shape and dtype, returning
// a new tensor. f32 only for now.
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);

// Activations, elementwise over f32. gelu is the tanh approximation used by
// GPT-2; silu (x * sigmoid(x)) is the one in LLaMA's feed-forward layers.
Tensor silu(const Tensor& x);
Tensor gelu(const Tensor& x);

// Softmax over the last dimension, f32. -inf entries (masks) become 0.
Tensor softmax(const Tensor& x);

// RMSNorm over the last dimension, as in LLaMA: x / sqrt(mean(x^2) + eps) * weight,
// where weight has the shape [last dimension of x]. f32 only.
Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps);

// A contiguous copy of x with its dimensions reordered: output dimension i is
// input dimension dims[i]. Up to 6 dimensions, any 32-bit dtype.
Tensor permute(const Tensor& x, std::vector<int> dims);

// permute that swaps two dimensions; negative dimensions count from the end.
Tensor transpose(const Tensor& x, int dim0, int dim1);

// Looks up rows of an f32 [vocab, dim] table for i32 ids of any shape, giving
// [ids..., dim]. Ids outside [0, vocab) give rows of zeros.
Tensor embedding(const Tensor& table, const Tensor& ids);

// Which elements of a head vector rotary embedding pairs up.
enum class RopeStyle : std::uint8_t {
    Interleaved,  // (2i, 2i + 1): Meta's LLaMA checkpoints and GGUF files
    RotateHalf,   // (i, i + dim / 2): HF transformers and GPT-NeoX
};

// cos and sin of position * theta^(-2i / head_dim) for every position below
// max_positions, as an f32 [max_positions, head_dim / 2, 2] tensor. Computed
// once on the host in double precision, because GPU sin and cos are only
// accurate near zero and these angles reach thousands of radians.
Tensor rope_table(Context& context, std::int64_t max_positions, std::int64_t head_dim, float theta);

// Rotary position embedding of x [..., seq, heads, head_dim], f32, where
// sequence index s sits at position start_pos + s. table comes from rope_table.
Tensor rope(const Tensor& x, const Tensor& table, std::int64_t start_pos, RopeStyle style);

// Matrix product of a [..., M, K] and b, f32 only. A 2-D b [K, N] is shared:
// leading dimensions of a fold into rows, the shape of a linear layer. A
// higher-rank b [..., K, N] must have exactly a's batch dimensions and is
// multiplied batch by batch. The result is [..., M, N].
Tensor matmul(const Tensor& a, const Tensor& b);

// As matmul, but with b stored transposed: [N, K] or [..., N, K]. This is a
// linear layer with PyTorch-layout weights [out, in], and attention scores
// q k^T, without materializing a transpose.
Tensor matmul_transposed(const Tensor& a, const Tensor& b);

}  // namespace vkml
