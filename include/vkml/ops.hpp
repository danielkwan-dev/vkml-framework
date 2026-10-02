#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "vkml/context.hpp"
#include "vkml/tensor.hpp"

namespace vkml {

// Elementwise arithmetic on two tensors of the same dtype, returning a new
// tensor of a's shape. b has a's shape, or its trailing dimensions, and then
// repeats over the leading ones: add(x [t, n], bias [n]) adds bias to every
// row. f32 only for now.
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);

// Activations, elementwise over f32. gelu is the tanh approximation used by
// GPT-2; silu (x * sigmoid(x)) is the one in LLaMA's feed-forward layers.
Tensor silu(const Tensor& x);
Tensor gelu(const Tensor& x);
// cap * tanh(x / cap), elementwise over f32: values squeezed smoothly into
// (-cap, cap), as Gemma 2 caps its logits (at 30). cap must be positive.
Tensor softcap(const Tensor& x, float cap);

// Softmax over the last dimension, f32. -inf entries (masks) become 0.
Tensor softmax(const Tensor& x);

// RMSNorm over the last dimension, as in LLaMA: x / sqrt(mean(x^2) + eps) * weight,
// where weight has the shape [last dimension of x]. f32 only.
Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps);

// x converted to dtype, in fresh storage. Widening f16 or bf16 to f32 is
// exact; f32 to f32 is a copy; narrowing f32 to f16 or bf16 rounds to
// nearest even. Other conversions are not implemented yet.
Tensor cast(const Tensor& x, DType dtype);

// A contiguous copy of x with its dimensions reordered: output dimension i is
// input dimension dims[i]. Up to 6 dimensions, any 32-bit dtype.
Tensor permute(const Tensor& x, std::vector<int> dims);

// permute that swaps two dimensions; negative dimensions count from the end.
Tensor transpose(const Tensor& x, int dim0, int dim1);

// Looks up rows of a [vocab, dim] table (f32, f16 or bf16) for i32 ids of any
// shape, giving f32 [ids..., dim]. Ids outside [0, vocab) give rows of zeros.
Tensor embedding(const Tensor& table, const Tensor& ids);

// Which elements of a head vector rotary embedding pairs up.
enum class RopeStyle : std::uint8_t {
    Interleaved,  // (2i, 2i + 1): Meta's LLaMA checkpoints and GGUF files
    RotateHalf,   // (i, i + dim / 2): HF transformers and GPT-NeoX
};

// LLaMA 3.1's rope scaling (rope_type "llama3"), which stretches the
// low frequencies so a model trained on original_max_positions tokens handles
// longer contexts: frequencies whose wavelength exceeds original_max_positions
// / low_freq_factor are divided by factor, those shorter than
// original_max_positions / high_freq_factor are kept, and those between are
// blended linearly in the ratio of context to wavelength.
struct RopeScaling {
    float factor;
    float low_freq_factor;
    float high_freq_factor;
    std::int64_t original_max_positions;
};

// cos and sin of position * theta^(-2i / head_dim) (with scaling applied to
// each frequency, if given, and each divided by freq_factors[i], if given, as
// llama.cpp stores LLaMA 3.1's scaling in GGUF files) for every position below
// max_positions, as an f32 [max_positions, head_dim / 2, 2] tensor. Computed
// once on the host in double precision, because GPU sin and cos are only
// accurate near zero and these angles reach thousands of radians.
// magnitude scales every cos and sin (LongRoPE's attention factor). A table
// for fewer dimensions than x's heads rotates only their first ones (Phi-3's
// partial rotary): build it with head_dim the rotated width.
Tensor rope_table(Context& context, std::int64_t max_positions, std::int64_t head_dim, float theta,
                  std::optional<RopeScaling> scaling = std::nullopt,
                  std::span<const float> freq_factors = {}, float magnitude = 1.0f);

// Rotary position embedding of x [..., seq, heads, head_dim], f32, where
// sequence index s sits at position start_pos + s. table comes from rope_table.
Tensor rope(const Tensor& x, const Tensor& table, std::int64_t start_pos, RopeStyle style);

// Matrix product of a [..., M, K] and b, giving f32. a is f32; b may be f32,
// f16 or bf16, so weights can stay 16-bit and halve memory and the bytes read
// per token, while arithmetic stays f32. A 2-D b [K, N] is shared:
// leading dimensions of a fold into rows, the shape of a linear layer. A
// higher-rank b [..., K, N] must have exactly a's batch dimensions and is
// multiplied batch by batch. The result is [..., M, N].
Tensor matmul(const Tensor& a, const Tensor& b);

// As matmul, but with b stored transposed: [N, K] or [..., N, K]. This is a
// linear layer with PyTorch-layout weights [out, in], and attention scores
// q k^T, without materializing a transpose.
Tensor matmul_transposed(const Tensor& a, const Tensor& b);

// The formats of llama.cpp that QuantizedMatrix holds. Each splits each row
// into blocks of 32 values sharing one f16 scale d, rounded to f16 before q
// is chosen, and stores integers q with the value q * d:
// - q8_0: q from -127 to 127 (a byte), d = max |x| / 127; 1.0625 bytes per
//   weight.
// - q4_0: q from -8 to 7 (four bits), d = m / -8 for m the block's value of
//   largest magnitude; 0.5625 bytes per weight.
// - q4_1: q from 0 to 15 with an f16 offset per block too, the value q * d +
//   min, d = (max - min) / 15, so both ends of each block are close; 0.625
//   bytes per weight. It also holds llama.cpp's q4_K, whose scales round to
//   f16.
// Decoding reads every weight once per token, so smaller weights run faster:
// against bf16's 2 bytes, q8_0 loses little accuracy, q4_0 noticeably more.
enum class QuantType : std::uint8_t { q8_0, q4_0, q4_1 };

// A weight matrix [rows, cols] in a quantized format. Scales (and q4_1's
// offsets) are f16, as llama.cpp keeps them; kernels read f32 ones too, as
// activations quantized on the fly have. values packs q, a row's
// values in order, 4 (q8_0) or 8 (q4_0, q4_1) to an i32 element: q8_0 a byte
// each, low bits first; q4_0 as q + 8 and q4_1 as q, value j of 8 in bits
// 8 (j % 4) + 4 (j / 4).
struct QuantizedMatrix {
    Tensor values;  // i32 [rows, cols / 4] (q8_0) or [rows, cols / 8] (q4_0, q4_1)
    Tensor scales;  // f16 or f32 [rows, cols / 32], or [rows, cols / 32, 2] (scale, min) for q4_1
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    QuantType type = QuantType::q8_0;
};

// Quantize an f32, f16 or bf16 matrix whose columns are a multiple of 32.
QuantizedMatrix quantize_q8(const Tensor& w);
QuantizedMatrix quantize_q4(const Tensor& w);
QuantizedMatrix quantize_q4_1(const Tensor& w);

// The f32 matrix the quantized values stand for.
Tensor dequantize(const QuantizedMatrix& q);

// As embedding, from a quantized table: each row decoded as dequantize would.
// A table too large for one GPU buffer as f16 (Gemma 3 4B's 262144 x 2560)
// fits quantized, and can share its memory with a tied output projection.
Tensor embedding(const QuantizedMatrix& table, const Tensor& ids);

// a [..., M, cols] times b transposed, giving [..., M, rows]: a linear layer
// with quantized weights.
Tensor matmul_transposed(const Tensor& a, const QuantizedMatrix& b);

// Scaled dot-product attention, softmax(q k^T / sqrt(head_dim)) v, per head.
// q is [heads, q_len, head_dim]; k and v are [kv_heads, kv_len, head_dim],
// where heads is a multiple of kv_heads (grouped-query attention shares each
// KV head among heads / kv_heads query heads). Query i sits at position
// i + kv_len - q_len, so new queries follow a KV cache; with causal, each
// attends only to keys at or before its position, and with a window (causal
// only) only to the last window of those: a query at position p sees keys
// p - window + 1 .. p, as Mistral's sliding window. With softcap > 0, each
// scaled score s becomes softcap * tanh(s / softcap) before the softmax, as
// Gemma 2 caps them (at 50). Returns [heads, q_len, head_dim].
Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, bool causal,
                 std::int64_t window = 0, float softcap = 0.0f);

}  // namespace vkml
