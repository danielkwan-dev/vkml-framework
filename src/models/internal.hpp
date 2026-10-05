#pragma once

// What the model's translation units share: where weights come from, and the
// config a GGUF file describes.

#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>

#include "vkml/llama.hpp"

namespace vkml {

namespace detail {
class Gguf;
}

// Where a Llama's weights come from, by their names in HF checkpoints
// ("model.layers.0.self_attn.q_proj.weight"). Vectors (norms, biases) come as
// f32, matrices as stored: 16-bit floats, or already quantized.
class detail::LlamaWeightSource {
public:
    virtual ~LlamaWeightSource() = default;
    virtual bool contains(const std::string& name) const = 0;
    // A float tensor of the given shape; throws if missing or shaped otherwise.
    virtual Tensor tensor(const std::string& name, const Shape& shape) const = 0;
    // The matrix as the file quantizes it, or as Q8_0 if it holds floats too
    // many for one GPU buffer; otherwise nothing.
    virtual std::optional<QuantizedMatrix> quantized(const std::string& name,
                                                     const Shape& shape) const = 0;
};

namespace detail {

// The weights of a GGUF file as llama.cpp's converter writes it. With
// sandwich_norms (Gemma), its ffn_norm is the MLP's input norm.
std::unique_ptr<LlamaWeightSource> gguf_weights(Context& context, const Gguf& file,
                                                bool sandwich_norms);

// The weights of HF safetensors shards.
std::unique_ptr<LlamaWeightSource> safetensors_weights(Context& context,
                                                       std::span<const SafeTensors> shards);

// A LLaMA-architecture model's hyperparameters from GGUF metadata, keyed by
// its architecture ("llama.block_count"): arch "llama" (LLaMA, Mistral,
// TinyLlama, SmolLM), "smollm3", "qwen2", "qwen3", "gemma2", "gemma3", "olmo2",
// "granite" or "phi3".
LlamaConfig config_from_gguf(const Gguf& file, const std::filesystem::path& path);

}  // namespace detail

}  // namespace vkml
