#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "io/gguf.hpp"
#include "models/internal.hpp"

namespace vkml {

LlamaConfig detail::config_from_gguf(const Gguf& file, const std::filesystem::path& path) {
    const auto& m = file.metadata();
    const std::string where = "Llama: " + path.string();
    const std::string arch = m.value("general.architecture", std::string("?"));
    if (arch != "llama" && arch != "smollm3" && arch != "qwen2" && arch != "qwen3" &&
        arch != "gemma2" && arch != "gemma3" && arch != "olmo2" && arch != "granite" &&
        arch != "phi3") {
        throw Error(where + " is a " + arch +
                    " model, which vkml does not implement (llama, smollm3, qwen2, qwen3, "
                    "gemma2, gemma3, olmo2, granite and phi3 are)");
    }
    const auto key = [&](const std::string& k) { return arch + "." + k; };
    const auto need = [&](const std::string& k) {
        if (!m.contains(key(k))) throw Error(where + " has no " + key(k));
        return m.at(key(k)).get<std::int64_t>();
    };
    LlamaConfig c;
    if (const std::string scaling = m.value(key("rope.scaling.type"), std::string("none"));
        scaling == "linear") {
        c.rope_linear_factor = m.value(key("rope.scaling.factor"), 1.0f);
    } else if (scaling != "none") {
        throw Error(where + " scales rope (" + scaling +
                    "), which vkml does not implement for GGUF files (linear it does)");
    }
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
    // converter reorders their rows to match (SmolLM3's and Granite's are
    // LLaMA's); Qwen's it leaves as HF has them.
    c.rope_style = arch == "llama" || arch == "smollm3" || arch == "granite"
                       ? RopeStyle::Interleaved
                       : RopeStyle::RotateHalf;
    // SmolLM3: no rope in every fourth layer, which llama.cpp fixes rather than reads.
    if (arch == "smollm3") {
        for (std::int64_t i = 0; i < c.num_layers; ++i) c.rope_layers.push_back((i + 1) % 4 != 0);
    }
    c.qkv_bias = file.contains("blk.0.attn_q.bias");
    c.qk_norm = file.contains("blk.0.attn_q_norm.weight");
    // Phi-3: part of each head rotates, and LongRoPE's factors are tensors.
    if (arch == "phi3") {
        if (const std::int64_t n = m.value(key("rope.dimension_count"), c.head_dim); n < c.head_dim)
            c.rotary_dim = n;
        if (file.contains("rope_factors_short.weight")) {
            c.longrope = LlamaConfig::LongRope{
                file.read_f32("rope_factors_short.weight"),
                file.read_f32("rope_factors_long.weight"),
                m.value(key("rope.scaling.original_context_length"), c.max_positions),
                m.value(key("rope.scaling.attn_factor"), 1.0f)};
        }
    }
    // Granite's multipliers; attention.scale replaces 1 / sqrt(head_dim).
    if (arch == "granite") {
        c.embedding_scale = m.value(key("embedding_scale"), 1.0f);
        if (const float s = m.value(key("attention.scale"), 0.0f); s > 0.0f)
            c.query_pre_attn_scalar = 1.0f / (s * s);
        c.residual_scale = m.value(key("residual_scale"), 1.0f);
        c.logit_divisor = m.value(key("logit_scale"), 1.0f);
    }
    // OLMo 2: norms on attention's and the MLP's outputs, none on their
    // inputs, and q and k normalized over their whole projections.
    if (arch == "olmo2") {
        c.sandwich_norms = true;
        c.pre_norms = false;
        c.qk_norm_whole = true;
    }
    // LLaMA 3.1's scaling, which llama.cpp's converter turns into a divisor
    // per frequency.
    if (file.contains("rope_freqs.weight"))
        c.rope_freq_factors = file.read_f32("rope_freqs.weight");
    if (m.contains("tokenizer.ggml.eos_token_id")) {
        c.eos_token_ids = {m.at("tokenizer.ggml.eos_token_id").get<std::int32_t>()};
    }
    // The file names one end-of-sequence token; chat turns may end with others,
    // which llama.cpp finds by name, as HF generation configs list them:
    // LLaMA 3's <|eom_id|> after tool calls, Phi-3's <|end|>, Gemma's
    // <end_of_turn>, Qwen's <|endoftext|> as well as <|im_end|>.
    {
        constexpr std::array<std::string_view, 6> kTurnEnds{
            "<|eot_id|>", "<|eom_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>"};
        const auto& tokens = m.value("tokenizer.ggml.tokens", nlohmann::json::array());
        for (std::size_t id = 0; id < tokens.size(); ++id) {
            if (!tokens[id].is_string()) continue;
            const auto& text = tokens[id].get_ref<const std::string&>();
            if (std::ranges::find(kTurnEnds, text) != kTurnEnds.end() &&
                std::ranges::find(c.eos_token_ids, std::int32_t(id)) == c.eos_token_ids.end()) {
                c.eos_token_ids.push_back(std::int32_t(id));
            }
        }
    }
    if (arch == "gemma2" || arch == "gemma3") {
        // What llama.cpp fixes for Gemma rather than reading. Gemma 2: every
        // other layer sliding, the first among them, and soft caps of 50 and
        // 30 unless the file says; Gemma 3: every sixth layer global, the rest
        // sliding at rope base 10000. Scores go over sqrt(head_dim), but for
        // the 27Bs (46 and 62 layers) hidden_size / heads. The converter has
        // already added the 1 to every norm weight.
        c.activation = Activation::gelu_tanh;
        c.embedding_scale = std::sqrt(float(c.hidden_size));
        c.sandwich_norms = true;
        if (m.contains(key("attention.sliding_window")))
            c.sliding_window = m.at(key("attention.sliding_window")).get<std::int64_t>();
        const bool gemma2 = arch == "gemma2";
        if (gemma2 && !c.sliding_window) c.sliding_window = 4096;
        for (std::int64_t i = 0; i < c.num_layers; ++i)
            c.sliding_layers.push_back(gemma2 ? i % 2 == 0 : (i + 1) % 6 != 0);
        if (gemma2) {
            c.attn_logit_softcap = m.value(key("attn_logit_softcapping"), 50.0f);
            c.final_logit_softcap = m.value(key("final_logit_softcapping"), 30.0f);
        } else {
            c.sliding_rope_theta = 10000.0f;
        }
        if (c.num_layers == (gemma2 ? 46 : 62))
            c.query_pre_attn_scalar = float(c.hidden_size / c.num_heads);
    }
    return c;
}

LlamaConfig LlamaConfig::from_json(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw Error("LlamaConfig: cannot open " + path.string());
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(in);
    } catch (const nlohmann::json::exception& e) {
        throw Error("LlamaConfig: " + path.string() + " is not valid JSON: " + e.what());
    }
    // Gemma 3 4B and up are image-text models: vkml runs the text model, whose
    // config is text_config, with the end-of-sequence tokens given outside it.
    if (json.value("model_type", "") == "gemma3" && json.contains("text_config")) {
        nlohmann::json text = json["text_config"];
        if (!text.contains("eos_token_id") && json.contains("eos_token_id"))
            text["eos_token_id"] = json["eos_token_id"];
        text["model_type"] = "gemma3_text";
        json = std::move(text);
    }
    // Gemma 3's configs leave out what they keep at Gemma3TextConfig's defaults.
    if (json.value("model_type", "") == "gemma3_text") {
        nlohmann::json defaults{{"vocab_size", 262208},
                                {"hidden_size", 2304},
                                {"intermediate_size", 9216},
                                {"num_hidden_layers", 26},
                                {"num_attention_heads", 8},
                                {"num_key_value_heads", 4},
                                {"head_dim", 256},
                                {"hidden_activation", "gelu_pytorch_tanh"},
                                {"max_position_embeddings", 131072},
                                {"rms_norm_eps", 1e-6},
                                {"rope_theta", 1000000.0},
                                {"rope_local_base_freq", 10000.0},
                                {"sliding_window", 4096},
                                {"sliding_window_pattern", 6},
                                {"query_pre_attn_scalar", 256}};
        defaults.update(json);
        json = std::move(defaults);
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
            if (kind == "linear") {
                c.rope_linear_factor = rs.at("factor").get<float>();
            } else if (kind == "llama3") {
                c.rope_scaling =
                    RopeScaling{rs.at("factor").get<float>(), rs.at("low_freq_factor").get<float>(),
                                rs.at("high_freq_factor").get<float>(),
                                rs.at("original_max_position_embeddings").get<std::int64_t>()};
            } else if (kind == "longrope") {
                c.longrope = LlamaConfig::LongRope{
                    rs.at("short_factor").get<std::vector<float>>(),
                    rs.at("long_factor").get<std::vector<float>>(),
                    rs.value("original_max_position_embeddings",
                             json.value("original_max_position_embeddings", std::int64_t{0})),
                    rs.value("attention_factor", 0.0f)};
            } else {
                throw Error("LlamaConfig: " + path.string() + " sets rope scaling of type " + kind +
                            ", which vkml does not implement (llama3, linear and longrope are)");
            }
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
        if (c.longrope && c.longrope->attention_factor == 0.0f) {
            // As transformers: from how far max_positions stretches the original.
            const double original = double(c.longrope->original_max_positions);
            const double factor = double(c.max_positions) / original;
            c.longrope->attention_factor =
                factor <= 1.0 ? 1.0f
                              : float(std::sqrt(1.0 + std::log(factor) / std::log(original)));
        }
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
        } else if (model_type == "gemma2") {
            // Gemma 3's layout before it: no q and k norms, one rope base,
            // every other layer sliding (the first among them), and soft caps
            // on the scores and the logits.
            c.embedding_scale = std::sqrt(float(c.hidden_size));
            c.norm_weight_offset = 1.0f;
            c.sandwich_norms = true;
            c.qkv_bias = c.o_bias = json.value("attention_bias", false);
            c.tie_word_embeddings = json.value("tie_word_embeddings", true);
            if (const auto& s = json.value("query_pre_attn_scalar", nlohmann::json{});
                s.is_number())
                c.query_pre_attn_scalar = s.get<float>();
            c.sliding_window = window();
            for (std::int64_t i = 0; i < c.num_layers; ++i) c.sliding_layers.push_back(i % 2 == 0);
            const auto cap = [&](const char* key) {
                const auto& v = json.value(key, nlohmann::json{});
                return v.is_number() ? v.get<float>() : 0.0f;
            };
            c.attn_logit_softcap = cap("attn_logit_softcapping");
            c.final_logit_softcap = cap("final_logit_softcapping");
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
            const auto& every = json.value("sliding_window_pattern", nlohmann::json{});
            const std::int64_t pattern = every.is_number() ? every.get<std::int64_t>() : 6;
            for (std::int64_t i = 0; i < c.num_layers; ++i)
                c.sliding_layers.push_back(pattern <= 0 || (i + 1) % pattern != 0);
            c.sliding_rope_theta = json.value("rope_local_base_freq", 10000.0f);
            if (params.is_object() && params.contains("full_attention")) {
                // Global layers may scale linearly (4B and up); sliding ones not at all.
                for (const char* kind : {"full_attention", "sliding_attention"}) {
                    const auto& p = params.at(kind);
                    const std::string type = p.value("rope_type", "default");
                    if (type == "linear" && std::string(kind) == "full_attention") {
                        c.rope_linear_factor = p.at("factor").get<float>();
                    } else if (type != "default") {
                        throw Error("LlamaConfig: " + path.string() + " sets rope_type " + type +
                                    " for " + kind + ", which vkml does not implement for Gemma");
                    }
                }
                c.rope_theta = params["full_attention"].value("rope_theta", c.rope_theta);
                c.sliding_rope_theta =
                    params["sliding_attention"].value("rope_theta", *c.sliding_rope_theta);
            }
        } else if (model_type == "phi3") {
            // Fused q, k, v and gate, up projections, split on loading.
            c.qkv_bias = c.o_bias = json.value("attention_bias", false);
            c.rotary_dim =
                std::int64_t(double(c.head_dim) * json.value("partial_rotary_factor", 1.0));
        } else if (model_type == "smollm3") {
            c.qkv_bias = c.o_bias = json.value("attention_bias", false);
            const auto& layers = json.value("no_rope_layers", nlohmann::json{});
            const std::int64_t every = json.value("no_rope_layer_interval", std::int64_t{4});
            for (std::int64_t i = 0; i < c.num_layers; ++i) {
                c.rope_layers.push_back(layers.is_array()
                                            ? layers.at(std::size_t(i)).get<int>() != 0
                                            : (i + 1) % every != 0);
            }
        } else if (model_type == "granite") {
            c.qkv_bias = c.o_bias = json.value("attention_bias", false);
            c.embedding_scale = json.value("embedding_multiplier", 1.0f);
            const float m = json.value("attention_multiplier", 0.0f);
            if (m > 0.0f) c.query_pre_attn_scalar = 1.0f / (m * m);
            c.residual_scale = json.value("residual_multiplier", 1.0f);
            c.logit_divisor = json.value("logits_scaling", 1.0f);
        } else if (model_type == "olmo2") {
            c.sandwich_norms = true;
            c.pre_norms = false;
            c.qk_norm = c.qk_norm_whole = true;
            c.qkv_bias = c.o_bias = json.value("attention_bias", false);
        } else if (model_type == "mistral") {
            c.sliding_window = window();  // null from Mistral 7B v0.2 on
        } else if (model_type == "llama") {
            c.qkv_bias = c.o_bias = json.value("attention_bias", false);
        } else {
            throw Error("LlamaConfig: " + path.string() + " has model_type " + model_type +
                        ", which vkml does not implement (llama, mistral, qwen2, qwen3, gemma2, "
                        "gemma3_text, olmo2, granite, smollm3 and phi3 are)");
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
    return detail::config_from_gguf(detail::Gguf{path}, path);
}

}  // namespace vkml
