#include "vkml/llama.hpp"

#include <algorithm>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "core/runtime.hpp"
#include "ops/internal.hpp"
#include "vkml/ops.hpp"

namespace vkml {

namespace {

// Layers recorded per submission during forward. Submitting early lets the GPU
// run each layer while the host records the next. On Iris Xe, TinyLlama decoded
// at 18.6 tokens/s with one submission per token and 22.0 with one per layer;
// 2, 4 and 8 layers per submission fell in between.
constexpr std::size_t kLayersPerSubmission = 1;

// Loads the named weight from whichever shard holds it, checking its shape.
// Matrices stay in the file's dtype, which matmul and embedding read directly;
// 16-bit weights halve memory and the bytes each decoding step reads. Vectors
// (norm weights) are small and widened to the f32 their ops take.
Tensor weight(Context& context, std::span<const SafeTensors> shards, const std::string& name,
              const Shape& shape) {
    for (const SafeTensors& shard : shards) {
        if (!shard.contains(name)) continue;
        if (shard.shape(name) != shape) {
            throw Error("Llama: weight " + name + " is " + to_string(shard.shape(name)) +
                        ", but the config needs " + to_string(shape));
        }
        const Tensor t = shard.load(context, name);
        return t.dtype() == DType::F32 || shape.size() > 1 ? t : cast(t, DType::F32);
    }
    throw Error("Llama: the checkpoint has no weight " + name);
}

// A loaded weight matrix, quantized when asked and its width allows.
std::variant<Tensor, QuantizedMatrix> matrix(const LlamaOptions& options, const Tensor& w) {
    if (options.quantize && w.shape().size() == 2 && w.shape()[1] % 32 == 0) {
        return *options.quantize == QuantType::q4_0 ? quantize_q4(w) : quantize_q8(w);
    }
    return w;
}

// The options for the output projection. At 4 bits it costs far more accuracy
// than any other matrix (SmolLM2 360M on wikitext: perplexity 9.83 against
// 8.67 with it at 8 bits, 7.64 unquantized), so it keeps 8, as llama.cpp's
// Q4_0 files keep it at 6 or more.
LlamaOptions head_options(LlamaOptions options) {
    if (options.quantize == QuantType::q4_0) options.quantize = QuantType::q8_0;
    return options;
}

DType validated_cache(DType dtype) {
    if (dtype != DType::F32 && dtype != DType::F16) {
        throw Error("Llama: a KV cache of " + std::string(to_string(dtype)) +
                    " is not implemented (f32 and f16 are)");
    }
    return dtype;
}

const LlamaConfig& validated(const LlamaConfig& c, std::int64_t context_length) {
    const bool positive = c.vocab_size > 0 && c.hidden_size > 0 && c.intermediate_size > 0 &&
                          c.num_layers > 0 && c.num_heads > 0 && c.num_kv_heads > 0 &&
                          c.head_dim > 0 && c.max_positions > 0;
    if (!positive || c.num_heads % c.num_kv_heads != 0 || c.head_dim % 2 != 0) {
        throw Error(
            "Llama: config needs positive sizes, num_heads a multiple of num_kv_heads, and an "
            "even head_dim");
    }
    if (context_length < 1 || context_length > c.max_positions) {
        throw Error("Llama: context length " + std::to_string(context_length) +
                    " must be between 1 and max_position_embeddings (" +
                    std::to_string(c.max_positions) + ")");
    }
    // A query at position p sees keys after p - window; within a context of
    // at most window positions that is every earlier key.
    if (c.sliding_window && context_length > *c.sliding_window) {
        throw Error("Llama: context length " + std::to_string(context_length) +
                    " exceeds the model's sliding window of " + std::to_string(*c.sliding_window) +
                    " tokens, which vkml does not implement; load it with a shorter context");
    }
    return c;
}

}  // namespace

LlamaConfig LlamaConfig::from_json(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw Error("LlamaConfig: cannot open " + path.string());
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(in);
    } catch (const nlohmann::json::exception& e) {
        throw Error("LlamaConfig: " + path.string() + " is not valid JSON: " + e.what());
    }

    LlamaConfig c;
    try {
        // transformers 5 writes rope_theta and any scaling in rope_parameters,
        // with rope_type "default" for none; earlier versions, at the top
        // level and in rope_scaling.
        const auto& params = json.value("rope_parameters", nlohmann::json{});
        nlohmann::json rs = json.value("rope_scaling", nlohmann::json{});
        if (rs.is_null() && params.is_object() &&
            params.value("rope_type", "default") != "default") {
            rs = params;
        }
        if (!rs.is_null()) {
            // HF has written the kind as "rope_type" and, earlier, "type".
            const std::string kind = rs.value("rope_type", rs.value("type", std::string("?")));
            if (kind != "llama3") {
                throw Error("LlamaConfig: " + path.string() + " sets rope scaling of type " + kind +
                            ", which vkml does not implement (llama3 is)");
            }
            c.rope_scaling =
                RopeScaling{rs.at("factor").get<float>(), rs.at("low_freq_factor").get<float>(),
                            rs.at("high_freq_factor").get<float>(),
                            rs.at("original_max_position_embeddings").get<std::int64_t>()};
        }
        c.vocab_size = json.at("vocab_size").get<std::int64_t>();
        c.hidden_size = json.at("hidden_size").get<std::int64_t>();
        c.intermediate_size = json.at("intermediate_size").get<std::int64_t>();
        c.num_layers = json.at("num_hidden_layers").get<std::int64_t>();
        c.num_heads = json.at("num_attention_heads").get<std::int64_t>();
        c.num_kv_heads = json.value("num_key_value_heads", c.num_heads);
        c.head_dim = json.value("head_dim", c.num_heads > 0 ? c.hidden_size / c.num_heads : 0);
        c.max_positions = json.at("max_position_embeddings").get<std::int64_t>();
        c.rms_norm_eps = json.value("rms_norm_eps", c.rms_norm_eps);
        c.rope_theta = json.value("rope_theta", c.rope_theta);
        if (params.is_object()) c.rope_theta = params.value("rope_theta", c.rope_theta);
        if (const std::string act = json.value("hidden_act", std::string("silu"));
            act == "gelu_pytorch_tanh" || act == "gelu_new") {
            c.activation = Activation::gelu_tanh;
        } else if (act != "silu" && act != "swish") {
            throw Error("LlamaConfig: " + path.string() + " sets hidden_act " + act +
                        ", which vkml does not implement (silu and gelu_pytorch_tanh are)");
        }
        c.tie_word_embeddings = json.value("tie_word_embeddings", false);
        const auto window = [&] {
            const auto& w = json.value("sliding_window", nlohmann::json{});
            return w.is_number() ? std::optional{w.get<std::int64_t>()} : std::nullopt;
        };
        const std::string model_type = json.value("model_type", std::string("llama"));
        if (model_type == "qwen2") {
            // Qwen2 configs carry a sliding_window that applies only when asked.
            if (json.value("use_sliding_window", false)) c.sliding_window = window();
            c.qkv_bias = true;
        } else if (model_type == "mistral") {
            c.sliding_window = window();  // null from Mistral 7B v0.2 on
        } else if (model_type == "llama") {
            c.qkv_bias = c.o_bias = json.value("attention_bias", false);
        } else {
            throw Error("LlamaConfig: " + path.string() + " has model_type " + model_type +
                        ", which vkml does not implement (llama, mistral and qwen2 are)");
        }
        if (json.value("mlp_bias", false)) {
            throw Error("LlamaConfig: " + path.string() +
                        " sets mlp_bias, which vkml does not implement");
        }
        if (const auto& eos = json.value("eos_token_id", nlohmann::json{}); eos.is_number()) {
            c.eos_token_ids = {eos.get<std::int32_t>()};
        } else if (eos.is_array()) {
            c.eos_token_ids = eos.get<std::vector<std::int32_t>>();
        }
    } catch (const nlohmann::json::exception& e) {
        throw Error("LlamaConfig: " + path.string() + ": " + e.what());
    }
    return c;
}

Llama Llama::load(Context& context, const std::filesystem::path& dir, std::int64_t context_length,
                  LlamaOptions options) {
    const LlamaConfig config = LlamaConfig::from_json(dir / "config.json");
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() == ".safetensors") files.push_back(entry.path());
    }
    if (files.empty()) throw Error("Llama: no .safetensors files in " + dir.string());
    std::ranges::sort(files);
    std::vector<SafeTensors> shards(files.begin(), files.end());
    return Llama{context, config, shards, context_length, options};
}

Llama::Llama(Context& context, LlamaConfig config, std::span<const SafeTensors> shards,
             std::int64_t context_length, LlamaOptions options)
    : context_(&context),
      config_(validated(config, context_length)),
      context_length_(context_length),
      embed_(weight(context, shards, "model.embed_tokens.weight",
                    {config.vocab_size, config.hidden_size})),
      final_norm_(weight(context, shards, "model.norm.weight", {config.hidden_size})),
      lm_head_(
          matrix(head_options(options), config.tie_word_embeddings
                                            ? embed_
                                            : weight(context, shards, "lm_head.weight",
                                                     {config.vocab_size, config.hidden_size}))),
      rope_table_(rope_table(context, context_length, config.head_dim, config.rope_theta,
                             config.rope_scaling)) {
    const std::int64_t d = config.hidden_size;
    const std::int64_t f = config.intermediate_size;
    const std::int64_t q_dim = config.num_heads * config.head_dim;
    const std::int64_t kv_dim = config.num_kv_heads * config.head_dim;
    const Shape cache_shape{config.num_kv_heads, context_length, config.head_dim};
    const DType cache_type = validated_cache(options.kv_cache);

    layers_.reserve(static_cast<std::size_t>(config.num_layers));
    for (std::int64_t l = 0; l < config.num_layers; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        const auto w = [&](const std::string& name, const Shape& shape) {
            return weight(context, shards, p + name, shape);
        };
        const auto m = [&](const std::string& name, const Shape& shape) {
            return matrix(options, w(name, shape));
        };
        const auto bias = [&](bool present, const std::string& name, std::int64_t size) {
            return present ? std::optional{w(name, {size})} : std::nullopt;
        };
        layers_.push_back(Layer{
            .input_norm = w("input_layernorm.weight", {d}),
            .q = m("self_attn.q_proj.weight", {q_dim, d}),
            .k = m("self_attn.k_proj.weight", {kv_dim, d}),
            .v = m("self_attn.v_proj.weight", {kv_dim, d}),
            .o = m("self_attn.o_proj.weight", {d, q_dim}),
            .q_bias = bias(config.qkv_bias, "self_attn.q_proj.bias", q_dim),
            .k_bias = bias(config.qkv_bias, "self_attn.k_proj.bias", kv_dim),
            .v_bias = bias(config.qkv_bias, "self_attn.v_proj.bias", kv_dim),
            .o_bias = bias(config.o_bias, "self_attn.o_proj.bias", d),
            .post_norm = w("post_attention_layernorm.weight", {d}),
            .gate = m("mlp.gate_proj.weight", {f, d}),
            .up = m("mlp.up_proj.weight", {f, d}),
            .down = m("mlp.down_proj.weight", {d, f}),
            .k_cache = Tensor::empty(context, cache_shape, cache_type),
            .v_cache = Tensor::empty(context, cache_shape, cache_type),
        });
    }
}

void Llama::rewind(std::int64_t position) {
    if (position < 0 || position > position_) {
        throw Error("Llama::rewind: cannot rewind to position " + std::to_string(position) +
                    " of a sequence at " + std::to_string(position_));
    }
    position_ = position;  // later cache rows are overwritten when those positions come again
}

Tensor Llama::forward(std::span<const std::int32_t> tokens) {
    const auto t = static_cast<std::int64_t>(tokens.size());
    if (t == 0) throw Error("Llama::forward: no tokens");
    if (position_ + t > context_length_) {
        throw Error("Llama::forward: " + std::to_string(position_ + t) +
                    " tokens exceed the context length of " + std::to_string(context_length_));
    }
    const LlamaConfig& c = config_;
    const std::int64_t hd = c.head_dim;
    const std::int64_t end = position_ + t;

    // x times w transposed. Projections of the same input share it, so that
    // it is quantized for quantized weights once (see detail::SharedInput).
    const auto project = [](detail::SharedInput& x, const Weight& w,
                            const std::optional<Tensor>& bias = std::nullopt) {
        Tensor y = std::visit([&](const auto& matrix) { return x.times_transposed(matrix); }, w);
        return bias ? add(y, *bias) : y;
    };

    // [seq, heads, head_dim] projections to [heads, seq, head_dim] for attention.
    const auto heads_first = [&](const Tensor& x, std::int64_t heads, bool rotate) {
        Tensor split = x.reshape({t, heads, hd});
        if (rotate) split = rope(split, rope_table_, position_, RopeStyle::RotateHalf);
        return permute(split, {1, 0, 2});
    };

    Tensor x = embedding(embed_, Tensor::from_data<std::int32_t>(*context_, tokens, {t}));
    // Submit every few layers so the GPU starts on them while the rest are
    // recorded, instead of idling until the whole forward pass is.
    std::size_t layer_index = 0;
    for (const Layer& layer : layers_) {
        detail::SharedInput h{rms_norm(x, layer.input_norm, c.rms_norm_eps)};
        const Tensor q = heads_first(project(h, layer.q, layer.q_bias), c.num_heads, true);
        // Appended to the caches in their type (rounded to f16, say).
        const auto append = [&](const Tensor& cache, const Tensor& rows) {
            detail::write_rows(
                cache, rows.dtype() == cache.dtype() ? rows : cast(rows, cache.dtype()), position_);
        };
        append(layer.k_cache, heads_first(project(h, layer.k, layer.k_bias), c.num_kv_heads, true));
        append(layer.v_cache,
               heads_first(project(h, layer.v, layer.v_bias), c.num_kv_heads, false));

        const Tensor attn = detail::attention(q, layer.k_cache, layer.v_cache, end, true);
        detail::SharedInput merged{permute(attn, {1, 0, 2}).reshape({t, c.num_heads * hd})};
        x = add(x, project(merged, layer.o, layer.o_bias));

        detail::SharedInput h2{rms_norm(x, layer.post_norm, c.rms_norm_eps)};
        const Tensor gate = project(h2, layer.gate);
        const Tensor activated = c.activation == Activation::gelu_tanh ? gelu(gate) : silu(gate);
        detail::SharedInput mlp{mul(activated, project(h2, layer.up))};
        x = add(x, project(mlp, layer.down));
        if (++layer_index % kLayersPerSubmission == 0) context_->runtime().stream.submit();
    }
    position_ = end;

    // Only the last position's logits are needed: gather its row first, since
    // the output projection is the largest matmul in the model.
    const Tensor last_id = Tensor::from_data<std::int32_t>(
        *context_, std::vector<std::int32_t>{static_cast<std::int32_t>(t - 1)}, {1});
    detail::SharedInput last{embedding(rms_norm(x, final_norm_, c.rms_norm_eps), last_id)};
    return project(last, lm_head_).reshape({c.vocab_size});
}

}  // namespace vkml
