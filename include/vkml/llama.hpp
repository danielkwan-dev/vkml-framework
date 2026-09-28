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

// The hyperparameters of a LLaMA-architecture model, as in HF's config.json.
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
    bool tie_word_embeddings = false;         // the output projection reuses the embedding table
    std::vector<std::int32_t> eos_token_ids;  // tokens that end generation; may be empty
    std::optional<RopeScaling> rope_scaling;  // rope_type "llama3" (LLaMA 3.1 and later)

    // Throws for settings vkml does not implement, such as rope_scaling other
    // than "llama3", rather than running the model wrongly.
    static LlamaConfig from_json(const std::filesystem::path& path);
};

struct LlamaOptions {
    // Quantize the weight matrices while loading (see QuantType): decoding
    // reads every weight per token, so smaller weights decode faster, for
    // some loss of accuracy. The embedding table and norm weights stay as
    // loaded, and so does any matrix whose width is not a multiple of 32.
    // With q4_0, the output projection, the most sensitive matrix, gets q8_0.
    std::optional<QuantType> quantize;
};

// A LLaMA-architecture decoder with HF-format weights, run in f32, with a KV
// cache for one sequence.
class Llama {
public:
    // Reads dir/config.json and every .safetensors file in dir. context_length
    // bounds the sequence length and sizes the KV cache; at most max_positions.
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

    struct Layer {
        Tensor input_norm;
        Weight q, k, v, o;
        Tensor post_norm;
        Weight gate, up, down;
        Tensor k_cache, v_cache;  // [num_kv_heads, context_length, head_dim]
    };

    Context* context_;
    LlamaConfig config_;
    std::int64_t context_length_;
    std::int64_t position_ = 0;
    Tensor embed_, final_norm_;
    Weight lm_head_;
    Tensor rope_table_;
    std::vector<Layer> layers_;
};

}  // namespace vkml
