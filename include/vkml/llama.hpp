#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <variant>
#include <vector>

#include "vkml/context.hpp"
#include "vkml/ops.hpp"
#include "vkml/safetensors.hpp"
#include "vkml/tensor.hpp"

namespace vkml {

namespace detail {
class LlamaWeightSource;  // safetensors shards or a GGUF file (src/models/llama.cpp)
}

// The feed-forward activation, applied to the gate projection: hidden_act
// "silu", or "gelu_pytorch_tanh" / "gelu_new", GELU's tanh approximation.
enum class Activation : std::uint8_t { silu, gelu_tanh };

// The hyperparameters of a LLaMA-architecture model, as in HF's config.json:
// model_type "llama", "mistral" (which adds a sliding window), "qwen2"
// (which adds biases), "qwen3" (which normalizes q and k per head),
// "gemma2" or "gemma3_text" (Gemma 2, and Gemma 3's text models; see below).
struct LlamaConfig {
    std::int64_t vocab_size = 0;
    std::int64_t hidden_size = 0;
    std::int64_t intermediate_size = 0;
    std::int64_t num_layers = 0;
    std::int64_t num_heads = 0;
    std::int64_t num_kv_heads = 0;  // fewer than num_heads: grouped-query attention
    std::int64_t head_dim = 0;
    std::int64_t max_positions = 0;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    bool tie_word_embeddings = false;  // the output projection reuses the embedding table
    Activation activation = Activation::silu;
    bool qkv_bias = false;  // biases on the q, k and v projections (Qwen2, or attention_bias)
    bool o_bias = false;    // and on the output projection (attention_bias)
    // RMSNorm on each head of q and k, before rope (Qwen3's q_norm and k_norm).
    bool qk_norm = false;
    std::vector<std::int32_t> eos_token_ids;  // tokens that end generation; may be empty
    std::optional<RopeScaling> rope_scaling;  // rope_type "llama3" (LLaMA 3.1 and later)
    // rope_type "linear": positions divided by this, as every frequency is
    // (Gemma 3 4B and up: 8, in their global layers only).
    float rope_linear_factor = 1.0f;
    // How q and k pair their elements for rope: HF checkpoints rotate halves;
    // llama.cpp's converter reorders LLaMA's q and k rows into interleaved pairs.
    RopeStyle rope_style = RopeStyle::RotateHalf;
    // Divisors of each rope frequency, as llama.cpp stores LLaMA 3.1's
    // scaling in GGUF files (rope_freqs.weight); empty for none.
    std::vector<float> rope_freq_factors;
    // Each query sees only the last this many keys, itself included (Mistral
    // 7B v0.1: 4096), in the layers sliding_layers marks, or in every layer
    // if it is empty.
    std::optional<std::int64_t> sliding_window;
    std::vector<bool> sliding_layers;  // empty, or one per layer
    // The rope base of the layers with the window, if not rope_theta
    // (Gemma 3's rope_local_base_freq).
    std::optional<float> sliding_rope_theta;

    // Gemma 3's differences. Embeddings are multiplied by embedding_scale
    // (sqrt(hidden_size)); every RMSNorm scales by norm_weight_offset +
    // weight (1 + weight, where GGUF files have the 1 added already); with
    // sandwich_norms, attention's and the MLP's outputs are normalized too
    // (post_attention_layernorm and post_feedforward_layernorm) before they
    // join the residual, and pre_feedforward_layernorm is the MLP's input
    // norm; and scores are q . k / sqrt(query_pre_attn_scalar) where it is set.
    float embedding_scale = 1.0f;
    float norm_weight_offset = 0.0f;
    bool sandwich_norms = false;
    std::optional<float> query_pre_attn_scalar;
    // Gemma 2's soft caps, cap * tanh(x / cap), on the scaled attention
    // scores (attn_logit_softcapping, 50) and on the output logits
    // (final_logit_softcapping, 30); 0 for none.
    float attn_logit_softcap = 0.0f;
    float final_logit_softcap = 0.0f;

    // Throws for settings vkml does not implement, such as rope_scaling other
    // than "llama3", exact GELU or another model_type,
    // rather than running the model wrongly. Reads rope settings from
    // rope_scaling and rope_theta, or rope_parameters (transformers 5).
    static LlamaConfig from_json(const std::filesystem::path& path);
    // The same from a GGUF file's metadata (see Llama::load).
    static LlamaConfig from_gguf(const std::filesystem::path& path);
};

struct LlamaOptions {
    // Quantize the weight matrices while loading (see QuantType): decoding
    // reads every weight per token, so smaller weights decode faster, for
    // some loss of accuracy. The embedding table and norm weights stay as
    // loaded, and so does any matrix whose width is not a multiple of 32.
    // With q4_0, the output projection, the most sensitive matrix, gets q8_0.
    std::optional<QuantType> quantize;

    // The KV cache's element type: f32, or f16 for half the memory (46 MB
    // instead of 92 for TinyLlama's 2048 positions) and half the bytes each
    // decoding step's attention reads, for rounding keys and values to 11
    // significant bits.
    DType kv_cache = DType::F32;
};

// A LLaMA-architecture decoder with HF-format weights, run in f32, with a KV
// cache for one sequence.
class Llama {
public:
    // Reads dir/config.json and every .safetensors file in dir, or, given a
    // .gguf file, its metadata and tensors (f32, f16, bf16, q8_0 and q4_0).
    // context_length bounds the sequence length and sizes the KV cache; at
    // most max_positions.
    static Llama load(Context& context, const std::filesystem::path& dir,
                      std::int64_t context_length, LlamaOptions options = {});

    Llama(Context& context, LlamaConfig config, std::span<const SafeTensors> shards,
          std::int64_t context_length, LlamaOptions options = {});

    const LlamaConfig& config() const noexcept { return config_; }
    std::int64_t context_length() const noexcept { return context_length_; }

    // Tokens processed so far in the current sequence.
    std::int64_t position() const noexcept { return position_; }

    // Runs tokens after those already processed and returns the logits for
    // the last one, [vocab_size]. A prompt can go in all at once.
    Tensor forward(std::span<const std::int32_t> tokens);

    // Starts a new sequence; the KV cache is overwritten as it goes.
    void reset() noexcept { position_ = 0; }

    // Forgets the tokens after the first position ones, so the sequence can
    // continue differently from there (a chat re-rendering its history, say).
    void rewind(std::int64_t position);

private:
    // A weight matrix, as loaded or quantized.
    using Weight = std::variant<Tensor, QuantizedMatrix>;

    Llama(Context& context, LlamaConfig config, const detail::LlamaWeightSource& weights,
          std::int64_t context_length, LlamaOptions options);

    struct Layer {
        Tensor input_norm;
        Weight q, k, v, o;
        std::optional<Tensor> q_bias, k_bias, v_bias, o_bias;
        std::optional<Tensor> q_norm, k_norm;  // [head_dim], with qk_norm
        Tensor post_norm;
        std::optional<Tensor> pre_ff_norm, post_ff_norm;  // with sandwich_norms
        Weight gate, up, down;
        Tensor k_cache, v_cache;  // [num_kv_heads, context_length, head_dim], kv_cache dtype
    };

    Context* context_;
    LlamaConfig config_;
    std::int64_t context_length_;
    std::int64_t position_ = 0;
    Weight embed_;  // a GGUF file's quantized table stays quantized
    Tensor final_norm_;
    Weight lm_head_;  // tied to a quantized embed_, the same buffers
    Tensor rope_table_;
    std::optional<Tensor> sliding_rope_table_;  // with sliding_rope_theta
    std::optional<Tensor> embedding_scale_;     // [hidden_size], unless embedding_scale is 1
    // [head_dim]: q's scale for query_pre_attn_scalar, without q norms to fold it into.
    std::optional<Tensor> q_scale_;
    std::vector<Layer> layers_;
};

}  // namespace vkml
