#include "vkml/llama.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "core/runtime.hpp"
#include "io/gguf.hpp"
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
}  // namespace

// Where a Llama's weights come from, by their names in HF checkpoints
// ("model.layers.0.self_attn.q_proj.weight"). Vectors (norms, biases) come as
// f32, matrices as stored: 16-bit floats, or already quantized.
class detail::LlamaWeightSource {
public:
    virtual ~LlamaWeightSource() = default;
    virtual bool contains(const std::string& name) const = 0;
    // A float tensor of the given shape; throws if missing or shaped otherwise.
    virtual Tensor tensor(const std::string& name, const Shape& shape) const = 0;
    // The matrix as the file quantizes it, or nothing if it holds floats.
    virtual std::optional<QuantizedMatrix> quantized(const std::string& name,
                                                     const Shape& shape) const = 0;
};

namespace {

void check_shape(const std::string& name, const Shape& got, const Shape& want) {
    if (got != want) {
        throw Error("Llama: weight " + name + " is " + to_string(got) + ", but the config needs " +
                    to_string(want));
    }
}

// Vectors widen to f32; matrices stay 16-bit.
Tensor as_loaded(const Tensor& t, const Shape& shape) {
    return t.dtype() == DType::F32 || shape.size() > 1 ? t : cast(t, DType::F32);
}

// A GGUF file as llama.cpp's converter writes it: HF names map to its own.
class GgufWeights final : public detail::LlamaWeightSource {
public:
    // With sandwich_norms (Gemma), llama.cpp's ffn_norm is the MLP's input
    // norm, and attention's output norm is post_attention_norm.
    GgufWeights(Context& context, const detail::Gguf& file, bool sandwich_norms)
        : context_(context), file_(file), sandwich_norms_(sandwich_norms) {}

    // HF's name for a weight, in llama.cpp's scheme.
    std::string name_of(const std::string& hf) const {
        if (hf == "model.embed_tokens.weight") return "token_embd.weight";
        if (hf == "model.norm.weight") return "output_norm.weight";
        if (hf == "lm_head.weight") return "output.weight";
        static const std::vector<std::pair<std::string, std::string>> parts{
            {"input_layernorm.", "attn_norm."},    {"self_attn.q_proj.", "attn_q."},
            {"self_attn.k_proj.", "attn_k."},      {"self_attn.v_proj.", "attn_v."},
            {"self_attn.o_proj.", "attn_output."}, {"self_attn.q_norm.", "attn_q_norm."},
            {"self_attn.k_norm.", "attn_k_norm."}, {"post_attention_layernorm.", "ffn_norm."},
            {"mlp.gate_proj.", "ffn_gate."},       {"mlp.up_proj.", "ffn_up."},
            {"mlp.down_proj.", "ffn_down."}};
        static const std::vector<std::pair<std::string, std::string>> sandwich_parts{
            {"post_attention_layernorm.", "post_attention_norm."},
            {"pre_feedforward_layernorm.", "ffn_norm."},
            {"post_feedforward_layernorm.", "post_ffw_norm."}};
        const std::string prefix = "model.layers.";
        if (hf.starts_with(prefix)) {
            const std::size_t dot = hf.find('.', prefix.size());
            const std::string layer = hf.substr(prefix.size(), dot - prefix.size());
            const std::string rest = hf.substr(dot + 1);
            for (const auto* list : {sandwich_norms_ ? &sandwich_parts : nullptr, &parts}) {
                if (!list) continue;
                for (const auto& [from, to] : *list) {
                    if (rest.starts_with(from))
                        return "blk." + layer + "." + to + rest.substr(from.size());
                }
            }
        }
        return hf;
    }

    bool contains(const std::string& name) const override { return file_.contains(name_of(name)); }

    Tensor tensor(const std::string& name, const Shape& shape) const override {
        const std::string g = name_of(name);
        if (!file_.contains(g)) throw Error("Llama: the GGUF file has no tensor " + g);
        check_shape(g, file_.shape(g), shape);
        // An embedding table stored quantized widens to f16 for lookups.
        if (is_quantized(file_.type_name(g))) return file_.load_f16(context_, g);
        return as_loaded(file_.load(context_, g), shape);
    }

    std::optional<QuantizedMatrix> quantized(const std::string& name,
                                             const Shape& shape) const override {
        const std::string g = name_of(name);
        if (!file_.contains(g)) return std::nullopt;
        const std::string type = file_.type_name(g);
        if (!is_quantized(type)) return std::nullopt;
        check_shape(g, file_.shape(g), shape);
        // vkml's own formats, and those it holds exactly: q4_K's sub-blocks
        // are Q4_1's.
        if (type == "q8_0" || type == "q4_0" || type == "q4_1" || type == "q4_K") {
            return file_.load_quantized(context_, g);
        }
        // The others (q5_0, q5_1, q5_K, q6_K), decoded on the host, run as
        // Q8_0: finer than any of them.
        return quantize_q8(file_.load_f32(context_, g));
    }

    static bool is_quantized(const std::string& type) {
        return type != "f32" && type != "f16" && type != "bf16";
    }

private:
    Context& context_;
    const detail::Gguf& file_;
    bool sandwich_norms_;
};

// A LLaMA-architecture model's hyperparameters from GGUF metadata, keyed by
// its architecture ("llama.block_count"): arch "llama" (LLaMA, Mistral,
// TinyLlama, SmolLM), "qwen2", "qwen3" or "gemma3".
LlamaConfig config_from_gguf(const detail::Gguf& file, const std::filesystem::path& path) {
    const auto& m = file.metadata();
    const std::string where = "Llama: " + path.string();
    const std::string arch = m.value("general.architecture", std::string("?"));
    if (arch != "llama" && arch != "qwen2" && arch != "qwen3" && arch != "gemma3") {
        throw Error(where + " is a " + arch +
                    " model, which vkml does not implement (llama, qwen2, qwen3 and gemma3 are)");
    }
    const auto key = [&](const std::string& k) { return arch + "." + k; };
    const auto need = [&](const std::string& k) {
        if (!m.contains(key(k))) throw Error(where + " has no " + key(k));
        return m.at(key(k)).get<std::int64_t>();
    };
    if (const std::string scaling = m.value(key("rope.scaling.type"), std::string("none"));
        scaling != "none") {
        throw Error(where + " scales rope (" + scaling +
                    "), which vkml does not implement for GGUF files");
    }
    LlamaConfig c;
    c.hidden_size = need("embedding_length");
    c.intermediate_size = need("feed_forward_length");
    c.num_layers = need("block_count");
    c.num_heads = need("attention.head_count");
    c.num_kv_heads = m.value(key("attention.head_count_kv"), c.num_heads);
    c.head_dim = m.value(key("attention.key_length"),
                         c.num_heads > 0 ? c.hidden_size / c.num_heads : std::int64_t{0});
    c.max_positions = need("context_length");
    c.rms_norm_eps = m.value(key("attention.layer_norm_rms_epsilon"), c.rms_norm_eps);
    c.rope_theta = m.value(key("rope.freq_base"), c.rope_theta);
    if (!file.contains("token_embd.weight")) throw Error(where + " has no token_embd.weight");
    c.vocab_size = file.shape("token_embd.weight").at(0);
    c.tie_word_embeddings = !file.contains("output.weight");
    // llama.cpp rotates LLaMA's q and k in interleaved pairs, and its
    // converter reorders their rows to match; Qwen's it leaves as HF has them.
    c.rope_style = arch == "llama" ? RopeStyle::Interleaved : RopeStyle::RotateHalf;
    c.qkv_bias = file.contains("blk.0.attn_q.bias");
    c.qk_norm = file.contains("blk.0.attn_q_norm.weight");
    // LLaMA 3.1's scaling, which llama.cpp's converter turns into a divisor
    // per frequency.
    if (file.contains("rope_freqs.weight"))
        c.rope_freq_factors = file.read_f32("rope_freqs.weight");
    if (m.contains("tokenizer.ggml.eos_token_id")) {
        c.eos_token_ids = {m.at("tokenizer.ggml.eos_token_id").get<std::int32_t>()};
    }
    if (arch == "gemma3") {
        // What llama.cpp fixes for Gemma 3 rather than reading: every sixth
        // layer global, the rest sliding at rope base 10000, and scores over
        // sqrt(head_dim), but for 27B's (62 layers) hidden_size / heads. Its
        // converter has already added the 1 to every norm weight.
        c.activation = Activation::gelu_tanh;
        c.embedding_scale = std::sqrt(float(c.hidden_size));
        c.sandwich_norms = true;
        if (m.contains(key("attention.sliding_window")))
            c.sliding_window = m.at(key("attention.sliding_window")).get<std::int64_t>();
        for (std::int64_t i = 0; i < c.num_layers; ++i)
            c.sliding_layers.push_back((i + 1) % 6 != 0);
        c.sliding_rope_theta = 10000.0f;
        if (c.num_layers == 62) c.query_pre_attn_scalar = float(c.hidden_size / c.num_heads);
        // Chat turns end with <end_of_turn>, which llama.cpp finds by name.
        const auto& tokens = m.value("tokenizer.ggml.tokens", nlohmann::json::array());
        for (std::size_t id = 0; id < tokens.size(); ++id) {
            if (tokens[id] == "<end_of_turn>") c.eos_token_ids.push_back(std::int32_t(id));
        }
    }
    return c;
}

// HF's safetensors shards.
class SafeTensorsWeights final : public detail::LlamaWeightSource {
public:
    SafeTensorsWeights(Context& context, std::span<const SafeTensors> shards)
        : context_(context), shards_(shards) {}

    bool contains(const std::string& name) const override {
        return std::ranges::any_of(shards_, [&](const SafeTensors& s) { return s.contains(name); });
    }
    Tensor tensor(const std::string& name, const Shape& shape) const override {
        for (const SafeTensors& shard : shards_) {
            if (!shard.contains(name)) continue;
            check_shape(name, shard.shape(name), shape);
            return as_loaded(shard.load(context_, name), shape);
        }
        throw Error("Llama: the checkpoint has no weight " + name);
    }
    std::optional<QuantizedMatrix> quantized(const std::string&, const Shape&) const override {
        return std::nullopt;
    }

private:
    Context& context_;
    std::span<const SafeTensors> shards_;
};

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

// The output projection: lm_head, or with tied embeddings the embedding
// table, which a GGUF file may hold quantized.
std::variant<Tensor, QuantizedMatrix> output_projection(const detail::LlamaWeightSource& weights,
                                                        const LlamaConfig& c, const Tensor& embed,
                                                        const LlamaOptions& options) {
    const Shape shape{c.vocab_size, c.hidden_size};
    const std::string name = c.tie_word_embeddings ? "model.embed_tokens.weight" : "lm_head.weight";
    if (auto q = weights.quantized(name, shape)) return *std::move(q);
    return matrix(options, c.tie_word_embeddings ? embed : weights.tensor(name, shape));
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
    if (!c.sliding_layers.empty() && std::int64_t(c.sliding_layers.size()) != c.num_layers) {
        throw Error("Llama: sliding_layers marks " + std::to_string(c.sliding_layers.size()) +
                    " layers of " + std::to_string(c.num_layers));
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
        // Gemma names it hidden_activation.
        if (const std::string act =
                json.value("hidden_activation", json.value("hidden_act", std::string("silu")));
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
        // Qwen configs carry a sliding_window that applies only when asked,
        // and then only from layer max_window_layers on.
        const auto qwen_window = [&] {
            if (!json.value("use_sliding_window", false)) return;
            c.sliding_window = window();
            const std::int64_t from = json.value("max_window_layers", std::int64_t{0});
            for (std::int64_t i = 0; i < c.num_layers; ++i) c.sliding_layers.push_back(i >= from);
        };
        const std::string model_type = json.value("model_type", std::string("llama"));
        if (model_type == "qwen2") {
            qwen_window();
            c.qkv_bias = true;
        } else if (model_type == "qwen3") {
            // Qwen2 without the biases, and with q and k normalized per head.
            qwen_window();
            c.qkv_bias = c.o_bias = json.value("attention_bias", false);
            c.qk_norm = true;
        } else if (model_type == "gemma3_text") {
            for (const char* key : {"attn_logit_softcapping", "final_logit_softcapping"}) {
                if (!json.value(key, nlohmann::json{}).is_null()) {
                    throw Error("LlamaConfig: " + path.string() + " sets " + key +
                                ", which vkml does not implement");
                }
            }
            c.embedding_scale = std::sqrt(float(c.hidden_size));
            c.norm_weight_offset = 1.0f;
            c.sandwich_norms = true;
            c.qk_norm = true;
            c.qkv_bias = c.o_bias = json.value("attention_bias", false);
            c.tie_word_embeddings = json.value("tie_word_embeddings", true);
            if (const auto& s = json.value("query_pre_attn_scalar", nlohmann::json{});
                s.is_number())
                c.query_pre_attn_scalar = s.get<float>();
            // Every pattern-th layer is global, the rest slide, with their own
            // rope base: in transformers 5, rope_parameters per kind of layer.
            c.sliding_window = window();
            const std::int64_t pattern = json.value("sliding_window_pattern", std::int64_t{6});
            for (std::int64_t i = 0; i < c.num_layers; ++i)
                c.sliding_layers.push_back(pattern <= 0 || (i + 1) % pattern != 0);
            c.sliding_rope_theta = json.value("rope_local_base_freq", 10000.0f);
            if (params.is_object() && params.contains("full_attention")) {
                for (const char* kind : {"full_attention", "sliding_attention"}) {
                    const auto& p = params.at(kind);
                    if (p.value("rope_type", "default") != "default") {
                        throw Error("LlamaConfig: " + path.string() + " sets rope_type " +
                                    p.value("rope_type", "") + " for " + kind +
                                    ", which vkml does not implement for Gemma");
                    }
                }
                c.rope_theta = params["full_attention"].value("rope_theta", c.rope_theta);
                c.sliding_rope_theta =
                    params["sliding_attention"].value("rope_theta", *c.sliding_rope_theta);
            }
        } else if (model_type == "mistral") {
            c.sliding_window = window();  // null from Mistral 7B v0.2 on
        } else if (model_type == "llama") {
            c.qkv_bias = c.o_bias = json.value("attention_bias", false);
        } else {
            throw Error("LlamaConfig: " + path.string() + " has model_type " + model_type +
                        ", which vkml does not implement (llama, mistral, qwen2, qwen3 and "
                        "gemma3_text are)");
        }
        // transformers 5 names each layer's kind, which overrides the above.
        if (const auto& types = json.value("layer_types", nlohmann::json{}); types.is_array()) {
            c.sliding_layers.clear();
            for (const auto& type : types) {
                const std::string kind = type.get<std::string>();
                if (kind != "sliding_attention" && kind != "full_attention") {
                    throw Error("LlamaConfig: " + path.string() + " has a layer of type " + kind +
                                ", which vkml does not implement");
                }
                c.sliding_layers.push_back(kind == "sliding_attention");
            }
            if (c.sliding_window && !json.value("use_sliding_window", true)) {
                c.sliding_window.reset();  // as HF, which drops the window then
            }
            if (!c.sliding_window) c.sliding_layers.clear();
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

LlamaConfig LlamaConfig::from_gguf(const std::filesystem::path& path) {
    return config_from_gguf(detail::Gguf{path}, path);
}

Llama Llama::load(Context& context, const std::filesystem::path& dir, std::int64_t context_length,
                  LlamaOptions options) {
    if (dir.extension() == ".gguf" && std::filesystem::is_regular_file(dir)) {
        const detail::Gguf file{dir};
        LlamaConfig config = config_from_gguf(file, dir);
        const bool sandwich = config.sandwich_norms;
        return Llama{context, std::move(config), GgufWeights{context, file, sandwich},
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
    : Llama(context, std::move(config), SafeTensorsWeights{context, shards}, context_length,
            options) {}

Llama::Llama(Context& context, LlamaConfig config, const detail::LlamaWeightSource& weights,
             std::int64_t context_length, LlamaOptions options)
    : context_(&context),
      config_(validated(config, context_length)),
      context_length_(context_length),
      embed_(weights.tensor("model.embed_tokens.weight", {config.vocab_size, config.hidden_size})),
      final_norm_(norm_weight(context, weights, config, "model.norm.weight", config.hidden_size)),
      lm_head_(output_projection(weights, config, embed_, head_options(options))),
      rope_table_(rope_table(context, context_length, config.head_dim, config.rope_theta,
                             config.rope_scaling, config.rope_freq_factors)) {
    if (config.sliding_rope_theta) {
        // Gemma 3's sliding layers rotate at their own base, unscaled.
        sliding_rope_table_ =
            rope_table(context, context_length, config.head_dim, *config.sliding_rope_theta);
    }
    if (config.embedding_scale != 1.0f) {
        embedding_scale_ = constant(context, config.hidden_size, config.embedding_scale);
    }
    // Scores divide by sqrt(head_dim); dividing by sqrt(query_pre_attn_scalar)
    // instead scales q, which the q norm's weight can do for free.
    float q_scale = 1.0f;
    if (config.query_pre_attn_scalar && *config.query_pre_attn_scalar != float(config.head_dim)) {
        if (!config.qk_norm) {
            throw Error("Llama: query_pre_attn_scalar without q and k norms is not implemented");
        }
        q_scale = std::sqrt(float(config.head_dim) / *config.query_pre_attn_scalar);
    }
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
        const auto head_norm = [&](const std::string& name, float scale = 1.0f) {
            if (!config.qk_norm) return std::optional<Tensor>{};
            const Tensor n = norm(name, config.head_dim);
            return std::optional{scale == 1.0f ? n
                                               : mul(n, constant(context, config.head_dim, scale))};
        };
        const auto sandwich = [&](const std::string& name) {
            return config.sandwich_norms ? std::optional{norm(name, d)} : std::nullopt;
        };
        layers_.push_back(Layer{
            .input_norm = norm("input_layernorm.weight", d),
            .q = m("self_attn.q_proj.weight", {q_dim, d}),
            .k = m("self_attn.k_proj.weight", {kv_dim, d}),
            .v = m("self_attn.v_proj.weight", {kv_dim, d}),
            .o = m("self_attn.o_proj.weight", {d, q_dim}),
            .q_bias = bias(config.qkv_bias, "self_attn.q_proj.bias", q_dim),
            .k_bias = bias(config.qkv_bias, "self_attn.k_proj.bias", kv_dim),
            .v_bias = bias(config.qkv_bias, "self_attn.v_proj.bias", kv_dim),
            .o_bias = bias(config.o_bias, "self_attn.o_proj.bias", d),
            .q_norm = head_norm("self_attn.q_norm.weight", q_scale),
            .k_norm = head_norm("self_attn.k_norm.weight"),
            .post_norm = norm("post_attention_layernorm.weight", d),
            .pre_ff_norm = sandwich("pre_feedforward_layernorm.weight"),
            .post_ff_norm = sandwich("post_feedforward_layernorm.weight"),
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

    // [seq, heads, head_dim] projections to [heads, seq, head_dim] for
    // attention, each head normalized (Qwen3) and rotated if asked.
    const auto heads_first = [&](const Tensor& x, std::int64_t heads, const Tensor* table,
                                 const std::optional<Tensor>& norm = std::nullopt) {
        Tensor split = x.reshape({t, heads, hd});
        if (norm) split = rms_norm(split, *norm, c.rms_norm_eps);
        if (table) split = rope(split, *table, position_, c.rope_style);
        return permute(split, {1, 0, 2});
    };

    Tensor x = embedding(embed_, Tensor::from_data<std::int32_t>(*context_, tokens, {t}));
    if (embedding_scale_) x = mul(x, *embedding_scale_);
    // Submit every few layers so the GPU starts on them while the rest are
    // recorded, instead of idling until the whole forward pass is.
    std::size_t layer_index = 0;
    for (const Layer& layer : layers_) {
        const bool slides =
            c.sliding_window && (c.sliding_layers.empty() || c.sliding_layers[layer_index]);
        const Tensor* table = slides && sliding_rope_table_ ? &*sliding_rope_table_ : &rope_table_;
        detail::SharedInput h{rms_norm(x, layer.input_norm, c.rms_norm_eps)};
        const Tensor q =
            heads_first(project(h, layer.q, layer.q_bias), c.num_heads, table, layer.q_norm);
        // Appended to the caches in their type (rounded to f16, say).
        const auto append = [&](const Tensor& cache, const Tensor& rows) {
            detail::write_rows(
                cache, rows.dtype() == cache.dtype() ? rows : cast(rows, cache.dtype()), position_);
        };
        append(layer.k_cache,
               heads_first(project(h, layer.k, layer.k_bias), c.num_kv_heads, table, layer.k_norm));
        append(layer.v_cache,
               heads_first(project(h, layer.v, layer.v_bias), c.num_kv_heads, nullptr));

        const Tensor attn = detail::attention(q, layer.k_cache, layer.v_cache, end, true,
                                              slides ? *c.sliding_window : 0);
        detail::SharedInput merged{permute(attn, {1, 0, 2}).reshape({t, c.num_heads * hd})};
        Tensor attn_out = project(merged, layer.o, layer.o_bias);
        // LLaMA normalizes the MLP's input with post_attention_layernorm;
        // Gemma, attention's output, and the MLP's input and output with the
        // feed-forward norms.
        if (c.sandwich_norms) attn_out = rms_norm(attn_out, layer.post_norm, c.rms_norm_eps);
        x = add(x, attn_out);

        detail::SharedInput h2{
            rms_norm(x, c.sandwich_norms ? *layer.pre_ff_norm : layer.post_norm, c.rms_norm_eps)};
        const Tensor gate = project(h2, layer.gate);
        const Tensor activated = c.activation == Activation::gelu_tanh ? gelu(gate) : silu(gate);
        detail::SharedInput mlp{mul(activated, project(h2, layer.up))};
        Tensor mlp_out = project(mlp, layer.down);
        if (c.sandwich_norms) mlp_out = rms_norm(mlp_out, *layer.post_ff_norm, c.rms_norm_eps);
        x = add(x, mlp_out);
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
