#include "vkml/llama.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "core/runtime.hpp"
#include "io/gguf.hpp"
#include "models/internal.hpp"
#include "ops/internal.hpp"
#include "vkml/ops.hpp"

namespace vkml {

namespace {

// Layers recorded per submission during forward. Submitting early lets the GPU
// run each layer while the host records the next. On Iris Xe, TinyLlama decoded
// at 18.6 tokens/s with one submission per token and 22.0 with one per layer;
// 2, 4 and 8 layers per submission fell in between.
constexpr std::size_t kLayersPerSubmission = 1;

// A loaded weight matrix, quantized when asked and its width allows.
std::variant<Tensor, QuantizedMatrix> matrix(const LlamaOptions& options, const Tensor& w) {
    if (options.quantize && w.shape().size() == 2 && w.shape()[1] % 32 == 0) {
        switch (*options.quantize) {
            case QuantType::q4_0: return quantize_q4(w);
            case QuantType::q4_1: return quantize_q4_1(w);
            case QuantType::q8_0: break;
        }
        return quantize_q8(w);
    }
    return w;
}

// The embedding table: as a GGUF file quantizes it, else as stored. Held
// quantized, it is a quarter to a half of f16's size, which Gemma 3 4B's
// 262144 x 2560 table needs to fit one GPU buffer.
std::variant<Tensor, QuantizedMatrix> embedding_table(const detail::LlamaWeightSource& weights,
                                                      const LlamaConfig& c) {
    const Shape shape{c.vocab_size, c.hidden_size};
    if (auto q = weights.quantized("model.embed_tokens.weight", shape)) return *std::move(q);
    return weights.tensor("model.embed_tokens.weight", shape);
}

// The output projection: lm_head, or with tied embeddings the embedding
// table itself, sharing its memory when it is quantized already.
std::variant<Tensor, QuantizedMatrix> output_projection(
    const detail::LlamaWeightSource& weights, const LlamaConfig& c,
    const std::variant<Tensor, QuantizedMatrix>& embed, const LlamaOptions& options) {
    if (c.tie_word_embeddings) {
        if (const auto* q = std::get_if<QuantizedMatrix>(&embed)) return *q;
        return matrix(options, std::get<Tensor>(embed));
    }
    const Shape shape{c.vocab_size, c.hidden_size};
    if (auto q = weights.quantized("lm_head.weight", shape)) return *std::move(q);
    return matrix(options, weights.tensor("lm_head.weight", shape));
}

// Each rope frequency's divisor: a GGUF file's (rope_freqs), times any
// linear scaling's factor; empty for none.
// The rotated width of each head.
std::int64_t rotary_dim(const LlamaConfig& c) { return c.rotary_dim ? c.rotary_dim : c.head_dim; }

// The divisor of each rope frequency, if any: LongRoPE's, for a context past
// the original or not (transformers switches when a sequence passes it).
std::vector<float> rope_divisors(const LlamaConfig& c, std::int64_t context_length) {
    if (c.longrope) {
        return context_length > c.longrope->original_max_positions ? c.longrope->long_factors
                                                                   : c.longrope->short_factors;
    }
    std::vector<float> d = c.rope_freq_factors;
    if (c.rope_linear_factor == 1.0f) return d;
    if (d.empty()) d.assign(std::size_t(c.head_dim / 2), 1.0f);
    for (float& f : d) f *= c.rope_linear_factor;
    return d;
}

// A [size] f32 tensor of value.
Tensor constant(Context& context, std::int64_t size, float value) {
    return Tensor::from_data<float>(context, std::vector<float>(std::size_t(size), value), {size});
}

// An RMSNorm weight, plus the config's offset (Gemma's 1 + weight).
Tensor norm_weight(Context& context, const detail::LlamaWeightSource& weights, const LlamaConfig& c,
                   const std::string& name, std::int64_t size) {
    const Tensor w = weights.tensor(name, {size});
    return c.norm_weight_offset == 0.0f ? w : add(w, constant(context, size, c.norm_weight_offset));
}

// The options for the output projection. At 4 bits it costs far more accuracy
// than any other matrix (SmolLM2 360M on wikitext: perplexity 9.83 against
// 8.67 with it at 8 bits, 7.64 unquantized), so it keeps 8, as llama.cpp's
// Q4_0 files keep it at 6 or more.
LlamaOptions head_options(LlamaOptions options) {
    if (options.quantize && *options.quantize != QuantType::q8_0)
        options.quantize = QuantType::q8_0;
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
    if (c.sliding_window && *c.sliding_window < 1) {
        throw Error("Llama: sliding window of " + std::to_string(*c.sliding_window) +
                    " tokens must be at least 1");
    }
    if (!c.rope_layers.empty() && std::int64_t(c.rope_layers.size()) != c.num_layers) {
        throw Error("Llama: rope_layers marks " + std::to_string(c.rope_layers.size()) +
                    " layers of " + std::to_string(c.num_layers));
    }
    if (!c.sliding_layers.empty() && std::int64_t(c.sliding_layers.size()) != c.num_layers) {
        throw Error("Llama: sliding_layers marks " + std::to_string(c.sliding_layers.size()) +
                    " layers of " + std::to_string(c.num_layers));
    }
    return c;
}

}  // namespace

Llama Llama::load(Context& context, const std::filesystem::path& dir, std::int64_t context_length,
                  LlamaOptions options) {
    if (!std::filesystem::exists(dir)) {
        throw Error("Llama::load: no model at " + dir.string() +
                    ": give a model directory (config.json and .safetensors files) or a .gguf "
                    "file");
    }
    if (dir.extension() == ".gguf" && std::filesystem::is_regular_file(dir)) {
        const detail::Gguf file{dir};
        LlamaConfig config = detail::config_from_gguf(file, dir);
        const bool sandwich = config.sandwich_norms;
        return Llama{context, std::move(config), *detail::gguf_weights(context, file, sandwich),
                     context_length, options};
    }
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
    : Llama(context, std::move(config), *detail::safetensors_weights(context, shards),
            context_length, options) {}

Llama::Llama(Context& context, LlamaConfig config, const detail::LlamaWeightSource& weights,
             std::int64_t context_length, LlamaOptions options)
    : context_(&context),
      config_(validated(config, context_length)),
      context_length_(context_length),
      embed_(embedding_table(weights, config)),
      final_norm_(norm_weight(context, weights, config, "model.norm.weight", config.hidden_size)),
      lm_head_(output_projection(weights, config, embed_, head_options(options))),
      rope_table_(rope_table(context, context_length, rotary_dim(config), config.rope_theta,
                             config.rope_scaling, rope_divisors(config, context_length),
                             config.longrope ? config.longrope->attention_factor : 1.0f)) {
    if (config.sliding_rope_theta) {
        // Gemma 3's sliding layers rotate at their own base, unscaled.
        sliding_rope_table_ =
            rope_table(context, context_length, config.head_dim, *config.sliding_rope_theta);
    }
    if (config.embedding_scale != 1.0f) {
        embedding_scale_ = constant(context, config.hidden_size, config.embedding_scale);
    }
    if (config.residual_scale != 1.0f) {
        residual_scale_ = constant(context, config.hidden_size, config.residual_scale);
    }
    if (config.logit_divisor != 1.0f) {
        logit_scale_ = constant(context, config.vocab_size, 1.0f / config.logit_divisor);
    }
    // Scores divide by sqrt(head_dim); dividing by sqrt(query_pre_attn_scalar)
    // instead scales q, which the q norm's weight can do for free (Gemma 3),
    // and a multiplication otherwise (Gemma 2 27B).
    float q_scale = 1.0f;
    if (config.query_pre_attn_scalar && *config.query_pre_attn_scalar != float(config.head_dim)) {
        q_scale = std::sqrt(float(config.head_dim) / *config.query_pre_attn_scalar);
        if (!config.qk_norm) q_scale_ = constant(context, config.head_dim, q_scale);
    }
    const std::int64_t d = config.hidden_size;
    const std::int64_t f = config.intermediate_size;
    const std::int64_t q_dim = config.num_heads * config.head_dim;
    const std::int64_t kv_dim = config.num_kv_heads * config.head_dim;
    const DType cache_type = validated_cache(options.kv_cache);
    if (options.sliding_extra_rows < 0) {
        throw Error("Llama: sliding_extra_rows must not be negative, got " +
                    std::to_string(options.sliding_extra_rows));
    }

    layers_.reserve(static_cast<std::size_t>(config.num_layers));
    for (std::int64_t l = 0; l < config.num_layers; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        const auto w = [&](const std::string& name, const Shape& shape) {
            return weights.tensor(p + name, shape);
        };
        const auto m = [&](const std::string& name, const Shape& shape) -> Weight {
            if (auto q = weights.quantized(p + name, shape)) return *std::move(q);
            return matrix(options, w(name, shape));
        };
        const auto bias = [&](bool present, const std::string& name, std::int64_t size) {
            return present ? std::optional{w(name, {size})} : std::nullopt;
        };
        const auto norm = [&](const std::string& name, std::int64_t size) {
            return norm_weight(context, weights, config, p + name, size);
        };
        const auto head_norm = [&](const std::string& name, std::int64_t whole,
                                   float scale = 1.0f) {
            if (!config.qk_norm) return std::optional<Tensor>{};
            const std::int64_t size = config.qk_norm_whole ? whole : config.head_dim;
            const Tensor n = norm(name, size);
            return std::optional{scale == 1.0f ? n : mul(n, constant(context, size, scale))};
        };
        const bool slides = config.sliding_window && (config.sliding_layers.empty() ||
                                                      config.sliding_layers[std::size_t(l)]);
        const std::int64_t rows =
            slides ? std::min(context_length, *config.sliding_window + options.sliding_extra_rows)
                   : context_length;
        const Shape cache_shape{config.num_kv_heads, rows, config.head_dim};
        const auto sandwich = [&](const std::string& name, bool pre = false) {
            return config.sandwich_norms && (!pre || config.pre_norms)
                       ? std::optional{norm(name, d)}
                       : std::nullopt;
        };
        layers_.push_back(Layer{
            .input_norm =
                config.pre_norms ? std::optional{norm("input_layernorm.weight", d)} : std::nullopt,
            .q = m("self_attn.q_proj.weight", {q_dim, d}),
            .k = m("self_attn.k_proj.weight", {kv_dim, d}),
            .v = m("self_attn.v_proj.weight", {kv_dim, d}),
            .o = m("self_attn.o_proj.weight", {d, q_dim}),
            .q_bias = bias(config.qkv_bias, "self_attn.q_proj.bias", q_dim),
            .k_bias = bias(config.qkv_bias, "self_attn.k_proj.bias", kv_dim),
            .v_bias = bias(config.qkv_bias, "self_attn.v_proj.bias", kv_dim),
            .o_bias = bias(config.o_bias, "self_attn.o_proj.bias", d),
            .q_norm = head_norm("self_attn.q_norm.weight", q_dim, q_scale),
            .k_norm = head_norm("self_attn.k_norm.weight", kv_dim),
            .post_norm = norm("post_attention_layernorm.weight", d),
            .pre_ff_norm = sandwich("pre_feedforward_layernorm.weight", true),
            .post_ff_norm = sandwich("post_feedforward_layernorm.weight"),
            .gate = m("mlp.gate_proj.weight", {f, d}),
            .up = m("mlp.up_proj.weight", {f, d}),
            .down = m("mlp.down_proj.weight", {d, f}),
            .k_cache = Tensor::empty(context, cache_shape, cache_type),
            .v_cache = Tensor::empty(context, cache_shape, cache_type),
            .slides = slides,
        });
    }
}

std::int64_t Llama::rewind(std::int64_t position) {
    if (position < 0 || position > position_) {
        throw Error("Llama::rewind: cannot rewind to position " + std::to_string(position) +
                    " of a sequence at " + std::to_string(position_));
    }
    // Later cache rows are overwritten when those positions come again; the
    // window of position's query must still be in the rings.
    const std::int64_t window = config_.sliding_window.value_or(1);
    if (std::max<std::int64_t>(0, position - window + 1) < oldest_kept_) position = 0;
    if (position == 0) oldest_kept_ = 0;
    position_ = position;
    return position;
}

std::int64_t Llama::kv_cache_bytes() const noexcept {
    std::int64_t bytes = 0;
    for (const Layer& layer : layers_) {
        bytes += std::int64_t(layer.k_cache.nbytes() + layer.v_cache.nbytes());
    }
    return bytes;
}

Tensor Llama::forward(std::span<const std::int32_t> tokens) { return run(tokens, false); }

Tensor Llama::forward_all(std::span<const std::int32_t> tokens) { return run(tokens, true); }

Tensor Llama::run(std::span<const std::int32_t> tokens, bool all_logits) {
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

    // [seq, heads, head_dim] projections to [heads, seq, head_dim] for
    // attention, each head normalized (Qwen3) and rotated if asked.
    const auto heads_first = [&](const Tensor& x, std::int64_t heads, const Tensor* table,
                                 const std::optional<Tensor>& norm = std::nullopt) {
        const bool whole = norm && c.qk_norm_whole;
        Tensor split = (whole ? rms_norm(x, *norm, c.rms_norm_eps) : x).reshape({t, heads, hd});
        if (norm && !whole) split = rms_norm(split, *norm, c.rms_norm_eps);
        if (table) split = rope(split, *table, position_, c.rope_style);
        return permute(split, {1, 0, 2});
    };

    const Tensor ids = Tensor::from_data<std::int32_t>(*context_, tokens, {t});
    Tensor x = std::visit([&](const auto& table) { return embedding(table, ids); }, embed_);
    if (embedding_scale_) x = mul(x, *embedding_scale_);
    // Submit every few layers so the GPU starts on them while the rest are
    // recorded, instead of idling until the whole forward pass is.
    std::size_t layer_index = 0;
    for (const Layer& layer : layers_) {
        const bool slides = layer.slides;
        const Tensor* table = slides && sliding_rope_table_ ? &*sliding_rope_table_ : &rope_table_;
        if (!c.rope_layers.empty() && !c.rope_layers[layer_index]) table = nullptr;
        detail::SharedInput h{layer.input_norm ? rms_norm(x, *layer.input_norm, c.rms_norm_eps)
                                               : x};
        Tensor q = heads_first(project(h, layer.q, layer.q_bias), c.num_heads, table, layer.q_norm);
        if (q_scale_) q = mul(q, *q_scale_);
        // Appended to the caches in their type (rounded to f16, say).
        const auto typed = [&](const Tensor& rows) {
            const DType type = layer.k_cache.dtype();
            return rows.dtype() == type ? rows : cast(rows, type);
        };
        const Tensor k = typed(
            heads_first(project(h, layer.k, layer.k_bias), c.num_kv_heads, table, layer.k_norm));
        const Tensor v =
            typed(heads_first(project(h, layer.v, layer.v_bias), c.num_kv_heads, nullptr));
        const Tensor attn = [&] {
            if (!slides) {
                detail::write_rows(layer.k_cache, k, position_);
                detail::write_rows(layer.v_cache, v, position_);
                return detail::attention(q, layer.k_cache, layer.v_cache, end, true, 0,
                                         detail::kMaxScoreBytes, c.attn_logit_softcap);
            }
            // A ring shorter than the window is the whole context: every key
            // it holds is in the window.
            const std::int64_t rows = layer.k_cache.shape()[1];
            oldest_kept_ = std::max(oldest_kept_, end - rows);
            return detail::sliding_attention(q, k, v, layer.k_cache, layer.v_cache, position_,
                                             std::min(*c.sliding_window, rows),
                                             detail::kMaxScoreBytes, c.attn_logit_softcap);
        }();
        detail::SharedInput merged{permute(attn, {1, 0, 2}).reshape({t, c.num_heads * hd})};
        Tensor attn_out = project(merged, layer.o, layer.o_bias);
        // LLaMA normalizes the MLP's input with post_attention_layernorm;
        // Gemma, attention's output, and the MLP's input and output with the
        // feed-forward norms.
        if (c.sandwich_norms) attn_out = rms_norm(attn_out, layer.post_norm, c.rms_norm_eps);
        if (residual_scale_) attn_out = mul(attn_out, *residual_scale_);
        x = add(x, attn_out);

        detail::SharedInput h2{
            !c.pre_norms ? x
                         : rms_norm(x, c.sandwich_norms ? *layer.pre_ff_norm : layer.post_norm,
                                    c.rms_norm_eps)};
        const Tensor gate = project(h2, layer.gate);
        const Tensor activated = c.activation == Activation::gelu_tanh ? gelu(gate) : silu(gate);
        detail::SharedInput mlp{mul(activated, project(h2, layer.up))};
        Tensor mlp_out = project(mlp, layer.down);
        if (c.sandwich_norms) mlp_out = rms_norm(mlp_out, *layer.post_ff_norm, c.rms_norm_eps);
        if (residual_scale_) mlp_out = mul(mlp_out, *residual_scale_);
        x = add(x, mlp_out);
        if (++layer_index % kLayersPerSubmission == 0) context_->runtime().stream.submit();
    }
    position_ = end;

    const Tensor normed = rms_norm(x, final_norm_, c.rms_norm_eps);
    const Tensor logits = [&] {
        if (all_logits) {
            detail::SharedInput rows{normed};
            return project(rows, lm_head_).reshape({t, c.vocab_size});
        }
        // Only the last position's logits are needed: gather its row first,
        // since the output projection is the largest matmul in the model.
        const Tensor last_id = Tensor::from_data<std::int32_t>(
            *context_, std::vector<std::int32_t>{static_cast<std::int32_t>(t - 1)}, {1});
        detail::SharedInput last{embedding(normed, last_id)};
        return project(last, lm_head_).reshape({c.vocab_size});
    }();
    if (logit_scale_) return mul(logits, *logit_scale_);
    return c.final_logit_softcap > 0.0f ? softcap(logits, c.final_logit_softcap) : logits;
}

}  // namespace vkml
