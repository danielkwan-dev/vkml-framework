#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <vkml/llama.hpp>
#include <vkml/vkml.hpp>

#include "support/gguf_writer.hpp"
#include "support/safetensors_writer.hpp"

using Catch::Matchers::ContainsSubstring;
using vkml::Llama;
using vkml::LlamaConfig;

namespace {

// LLaMA 3.1's rope scaling of one inverse frequency, transcribed from HF's
// _compute_llama3_parameters: long wavelengths are divided by factor, short
// ones kept, and the band between blended.
double llama3_inv_freq(double inv_freq, const vkml::RopeScaling& s) {
    const double pi = 3.141592653589793;
    const double old_len = double(s.original_max_positions);
    const double low_freq_wavelen = old_len / s.low_freq_factor;
    const double high_freq_wavelen = old_len / s.high_freq_factor;
    const double wavelen = 2 * pi / inv_freq;
    if (wavelen > low_freq_wavelen) return inv_freq / s.factor;
    if (wavelen < high_freq_wavelen) return inv_freq;
    const double smooth =
        (old_len / wavelen - s.low_freq_factor) / (s.high_freq_factor - s.low_freq_factor);
    return (1 - smooth) * inv_freq / s.factor + smooth * inv_freq;
}

// The architecture a TinyModel follows: LLaMA; Qwen2, which adds biases on the
// q, k and v projections; Qwen3, which instead RMS-normalizes each head of q
// and k before rope; or Gemma 3, which adds to that scaled embeddings, norms
// of 1 + weight, norms on attention's and the MLP's outputs, a sliding window
// with its own rope base, and a query scale of its own; or OLMo 2, which
// normalizes attention's and the MLP's outputs but not their inputs, and q
// and k over their whole projections; or Granite, LLaMA with multipliers on
// the embeddings, scores, residual branches and logits; or SmolLM3, LLaMA
// with no rope in every fourth layer (here 4 layers, the last without).
enum class Arch { llama, qwen2, qwen3, gemma2, gemma3, olmo2, granite, smollm3 };

// A tiny LLaMA with random weights: 2 layers, grouped-query attention with 2
// query heads per KV head, and an odd vocabulary size.
struct TinyModel {
    LlamaConfig config;
    std::map<std::string, std::vector<float>> weights;
    std::map<std::string, vkml::Shape> shapes;

    bool bf16 = false;  // weights stored as bf16, as most HF checkpoints are
    // Written as Gemma 3 4B and up are: an image-text model whose config nests
    // the text model's, and whose weights sit under language_model.
    bool image_text = false;

    Arch arch = Arch::llama;

    explicit TinyModel(bool tie_embeddings, bool bf16_weights = false, bool rope_scaled = false,
                       Arch architecture = Arch::llama)
        : bf16(bf16_weights), arch(architecture) {
        config.vocab_size = 37;
        config.hidden_size = 32;
        config.intermediate_size = 48;
        config.num_layers = 2;
        config.num_heads = 4;
        config.num_kv_heads = 2;
        // Qwen3's heads are wider than hidden_size / num_heads (128 against
        // 64 in Qwen3 0.6B), so q's projection is wider than the model.
        const bool gemma = arch == Arch::gemma2 || arch == Arch::gemma3;
        config.head_dim = arch == Arch::qwen3 || gemma ? 16 : 8;
        config.max_positions = 64;
        config.rms_norm_eps = 1e-5f;
        config.rope_theta = 10000.0f;
        config.tie_word_embeddings = tie_embeddings;
        // With head_dim 8 and theta 10000 the wavelengths are about 6, 63, 628
        // and 6283, so these settings keep one frequency, blend one and divide
        // two.
        if (rope_scaled) config.rope_scaling = vkml::RopeScaling{4.0f, 1.0f, 4.0f, 64};
        if (arch == Arch::gemma3) {
            // As write() puts it in config.json: a window in the first layer
            // (sliding_window_pattern 2), rotating at base 100.
            config.activation = vkml::Activation::gelu_tanh;
            config.sliding_window = 3;
            config.sliding_layers = {true, false};
            config.sliding_rope_theta = 100.0f;
            config.embedding_scale = std::sqrt(float(config.hidden_size));
            config.norm_weight_offset = 1.0f;
            config.sandwich_norms = true;
            config.qk_norm = true;
            config.query_pre_attn_scalar = 8.0f;
        }
        if (arch == Arch::smollm3) {
            config.num_layers = 4;
            config.rope_layers = {true, true, true, false};
        }
        if (arch == Arch::granite) {
            // Scores times 1/4 (attention_multiplier), as 1 / sqrt(16).
            config.embedding_scale = 3.0f;
            config.query_pre_attn_scalar = 16.0f;
            config.residual_scale = 0.5f;
            config.logit_divisor = 2.0f;
        }
        if (arch == Arch::olmo2) {
            config.sandwich_norms = true;
            config.pre_norms = false;
            config.qk_norm = config.qk_norm_whole = true;
        }
        if (arch == Arch::gemma2) {
            // Gemma 3 without q and k norms or a second rope base, and with
            // caps low enough to bend most scores and logits.
            config.activation = vkml::Activation::gelu_tanh;
            config.sliding_window = 3;
            config.sliding_layers = {true, false};
            config.embedding_scale = std::sqrt(float(config.hidden_size));
            config.norm_weight_offset = 1.0f;
            config.sandwich_norms = true;
            config.query_pre_attn_scalar = 8.0f;
            config.attn_logit_softcap = 1.0f;
            config.final_logit_softcap = 3.0f;
        }
        // Gemma's norm weights are offsets from 1.
        const float norm_mean = gemma ? 0.0f : 1.0f;

        std::mt19937 rng{1234};
        const auto add = [&](const std::string& name, vkml::Shape shape, float mean, float stddev) {
            std::normal_distribution<float> dist{mean, stddev};
            std::size_t n = 1;
            for (auto d : shape) n *= std::size_t(d);
            std::vector<float> v(n);
            for (float& x : v) x = dist(rng);
            weights[name] = std::move(v);
            shapes[name] = std::move(shape);
        };
        const std::int64_t d = config.hidden_size, f = config.intermediate_size;
        const std::int64_t q_dim = config.num_heads * config.head_dim;
        const std::int64_t kv_dim = config.num_kv_heads * config.head_dim;
        add("model.embed_tokens.weight", {config.vocab_size, d}, 0.0f, 1.0f);
        for (int l = 0; l < config.num_layers; ++l) {
            const std::string p = "model.layers." + std::to_string(l) + ".";
            if (arch != Arch::olmo2) add(p + "input_layernorm.weight", {d}, norm_mean, 0.1f);
            add(p + "self_attn.q_proj.weight", {q_dim, d}, 0.0f, 0.2f);
            add(p + "self_attn.k_proj.weight", {kv_dim, d}, 0.0f, 0.2f);
            add(p + "self_attn.v_proj.weight", {kv_dim, d}, 0.0f, 0.2f);
            add(p + "self_attn.o_proj.weight", {d, q_dim}, 0.0f, 0.2f);
            if (arch == Arch::qwen2) {
                add(p + "self_attn.q_proj.bias", {q_dim}, 0.0f, 0.5f);
                add(p + "self_attn.k_proj.bias", {kv_dim}, 0.0f, 0.5f);
                add(p + "self_attn.v_proj.bias", {kv_dim}, 0.0f, 0.5f);
            }
            if (arch == Arch::qwen3 || arch == Arch::gemma3) {
                add(p + "self_attn.q_norm.weight", {config.head_dim}, norm_mean, 0.3f);
                add(p + "self_attn.k_norm.weight", {config.head_dim}, norm_mean, 0.3f);
            }
            if (arch == Arch::olmo2) {
                add(p + "self_attn.q_norm.weight", {q_dim}, norm_mean, 0.3f);
                add(p + "self_attn.k_norm.weight", {kv_dim}, norm_mean, 0.3f);
                add(p + "post_feedforward_layernorm.weight", {d}, norm_mean, 0.1f);
            }
            add(p + "post_attention_layernorm.weight", {d}, norm_mean, 0.1f);
            if (gemma) {
                add(p + "pre_feedforward_layernorm.weight", {d}, norm_mean, 0.1f);
                add(p + "post_feedforward_layernorm.weight", {d}, norm_mean, 0.1f);
            }
            add(p + "mlp.gate_proj.weight", {f, d}, 0.0f, 0.2f);
            add(p + "mlp.up_proj.weight", {f, d}, 0.0f, 0.2f);
            add(p + "mlp.down_proj.weight", {d, f}, 0.0f, 0.2f);
        }
        add("model.norm.weight", {d}, norm_mean, 0.1f);
        if (!tie_embeddings) add("lm_head.weight", {config.vocab_size, d}, 0.0f, 0.2f);

        // Round to bf16 (truncating the low bits) so the reference computes
        // with exactly the values the file holds.
        if (bf16) {
            for (auto& [name, values] : weights) {
                for (float& v : values) {
                    std::uint32_t bits;
                    std::memcpy(&bits, &v, 4);
                    bits &= 0xFFFF0000u;
                    std::memcpy(&v, &bits, 4);
                }
            }
        }
    }

    // config.json plus the weights split over two shards, as HF saves them.
    std::filesystem::path write(const std::string& dir_name) const {
        const auto sliding_json = [&] {
            if (!config.sliding_window || arch == Arch::gemma2 || arch == Arch::gemma3)
                return std::string();
            std::string out = R"(, "sliding_window": )" + std::to_string(*config.sliding_window);
            if (config.sliding_layers.empty()) return out;
            out += R"(, "layer_types": [)";
            for (std::size_t i = 0; i < config.sliding_layers.size(); ++i) {
                out += i ? ", " : "";
                out += config.sliding_layers[i] ? R"("sliding_attention")" : R"("full_attention")";
            }
            return out + "]";
        };
        const auto dir = vkml_test::temp_path(dir_name);
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        std::ostringstream text;
        text
            << (arch == Arch::qwen2
                    ? R"({"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2",)"
                      R"( "use_sliding_window": false, "vocab_size": )"
                : arch == Arch::qwen3
                    ? R"({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",)"
                      R"( "attention_bias": false, "use_sliding_window": false, "vocab_size": )"
                : arch == Arch::gemma2
                    ? R"({"architectures": ["Gemma2ForCausalLM"], "model_type": "gemma2",)"
                      R"( "hidden_activation": "gelu_pytorch_tanh", "query_pre_attn_scalar": )" +
                          std::to_string(config.query_pre_attn_scalar.value_or(0.0f)) +
                          R"(, "sliding_window": )" + std::to_string(*config.sliding_window) +
                          R"(, "attn_logit_softcapping": )" +
                          std::to_string(config.attn_logit_softcap) +
                          R"(, "final_logit_softcapping": )" +
                          std::to_string(config.final_logit_softcap) + R"(, "vocab_size": )"
                : arch == Arch::gemma3
                    ? R"({"architectures": ["Gemma3ForCausalLM"], "model_type": "gemma3_text",)"
                      R"( "hidden_activation": "gelu_pytorch_tanh", "query_pre_attn_scalar": )" +
                          std::to_string(config.query_pre_attn_scalar.value_or(0.0f)) +
                          R"(, "sliding_window": )" + std::to_string(*config.sliding_window) +
                          R"(, "sliding_window_pattern": 2, "rope_local_base_freq": )" +
                          std::to_string(*config.sliding_rope_theta) +
                          R"(, "attn_logit_softcapping": null,)"
                          R"( "final_logit_softcapping": null, "vocab_size": )"
                : arch == Arch::olmo2
                    ? R"({"architectures": ["Olmo2ForCausalLM"], "model_type": "olmo2",)"
                      R"( "vocab_size": )"
                : arch == Arch::smollm3
                    ? R"({"architectures": ["SmolLM3ForCausalLM"], "model_type": "smollm3",)"
                      R"( "no_rope_layers": [1, 1, 1, 0], "no_rope_layer_interval": 4, "vocab_size": )"
                : arch == Arch::granite
                    ? R"({"architectures": ["GraniteForCausalLM"], "model_type": "granite",)"
                      R"( "embedding_multiplier": 3.0, "attention_multiplier": 0.25,)"
                      R"( "residual_multiplier": 0.5, "logits_scaling": 2.0, "vocab_size": )"
                : config.sliding_window
                    // Mistral, as it has a window, which layer_types can narrow.
                    ? R"({"architectures": ["MistralForCausalLM"], "model_type": "mistral",)"
                      R"( "vocab_size": )"
                    : R"({"architectures": ["LlamaForCausalLM"], "vocab_size": )")
            << config.vocab_size << sliding_json() << R"(, "head_dim": )" << config.head_dim
            << R"(, "hidden_size": )" << config.hidden_size << R"(, "intermediate_size": )"
            << config.intermediate_size << R"(, "num_hidden_layers": )" << config.num_layers
            << R"(, "num_attention_heads": )" << config.num_heads << R"(, "num_key_value_heads": )"
            << config.num_kv_heads << R"(, "max_position_embeddings": )" << config.max_positions
            << R"(, "rms_norm_eps": 1e-05, "rope_theta": 10000.0, "rope_scaling": )"
            << (config.rope_scaling
                    ? R"({"rope_type": "llama3", "factor": 4.0, "low_freq_factor": 1.0,)"
                      R"( "high_freq_factor": 4.0, "original_max_position_embeddings": 64})"
                : config.rope_linear_factor != 1.0f
                    ? R"({"rope_type": "linear", "factor": )" +
                          std::to_string(config.rope_linear_factor) + "}"
                    : std::string("null"))
            << ","
            << R"( "tie_word_embeddings": )" << (config.tie_word_embeddings ? "true" : "false")
            << R"(, "hidden_act": )"
            << (config.activation == vkml::Activation::gelu_tanh ? R"("gelu_pytorch_tanh")"
                                                                 : R"("silu")")
            << "}";
        std::ofstream(dir / "config.json")
            << (image_text ? R"({"architectures": ["Gemma3ForConditionalGeneration"],)"
                             R"( "model_type": "gemma3", "eos_token_id": [2, 7], "text_config": )" +
                                 text.str() + "}"
                           : text.str());

        std::vector<vkml_test::Entry> shard1, shard2;
        // A vision weight, which the text model must leave alone.
        if (image_text)
            shard1.push_back({"vision_tower.patch.weight",
                              "F32",
                              {2},
                              vkml_test::raw(std::vector<float>{1, 2})});
        for (const auto& [hf_name, values] : weights) {
            const std::string name = image_text ? "language_model." + hf_name : hf_name;
            auto& shard = name.find("layers.1.") != std::string::npos ? shard2 : shard1;
            if (bf16) {
                std::vector<std::uint16_t> halves(values.size());
                for (std::size_t i = 0; i < values.size(); ++i) {
                    std::uint32_t bits;
                    std::memcpy(&bits, &values[i], 4);
                    halves[i] = std::uint16_t(bits >> 16);
                }
                shard.push_back({name, "BF16", shapes.at(hf_name), vkml_test::raw(halves)});
            } else {
                shard.push_back({name, "F32", shapes.at(hf_name), vkml_test::raw(values)});
            }
        }
        vkml_test::write_safetensors(dir / "model-00001-of-00002.safetensors", shard1);
        vkml_test::write_safetensors(dir / "model-00002-of-00002.safetensors", shard2);
        return dir;
    }

    // The same model as llama.cpp's converter writes it to GGUF: its tensor
    // names, dimensions fastest first, and for LLaMA the rows of each head of
    // q and k reordered from HF's rotate-half pairs (i, i + d/2) to
    // interleaved ones (2i, 2i + 1), which llama.cpp's rope rotates. Qwen
    // models it leaves in HF's order, and rotates them as HF does.
    std::filesystem::path write_gguf(const std::string& file) const {
        const std::string a = arch == Arch::llama     ? "llama"
                              : arch == Arch::smollm3 ? "smollm3"
                              : arch == Arch::qwen2   ? "qwen2"
                              : arch == Arch::qwen3   ? "qwen3"
                              : arch == Arch::gemma2  ? "gemma2"
                                                      : "gemma3";
        const bool gemma = arch == Arch::gemma2 || arch == Arch::gemma3;
        vkml_test::GgufWriter g;
        g.string("general.architecture", a);
        g.u32(a + ".context_length", std::uint32_t(config.max_positions));
        g.u32(a + ".embedding_length", std::uint32_t(config.hidden_size));
        g.u32(a + ".block_count", std::uint32_t(config.num_layers));
        g.u32(a + ".feed_forward_length", std::uint32_t(config.intermediate_size));
        g.u32(a + ".attention.head_count", std::uint32_t(config.num_heads));
        g.u32(a + ".attention.head_count_kv", std::uint32_t(config.num_kv_heads));
        g.u32(a + ".attention.key_length", std::uint32_t(config.head_dim));
        g.u32(a + ".attention.value_length", std::uint32_t(config.head_dim));
        g.f32(a + ".attention.layer_norm_rms_epsilon", config.rms_norm_eps);
        g.f32(a + ".rope.freq_base", config.rope_theta);
        if (gemma) g.u32(a + ".attention.sliding_window", std::uint32_t(*config.sliding_window));
        if (arch == Arch::gemma2) {
            g.f32(a + ".attn_logit_softcapping", config.attn_logit_softcap);
            g.f32(a + ".final_logit_softcapping", config.final_logit_softcap);
        }
        if (config.rope_linear_factor != 1.0f) {
            g.string(a + ".rope.scaling.type", "linear");
            g.f32(a + ".rope.scaling.factor", config.rope_linear_factor);
        }
        g.u32("tokenizer.ggml.eos_token_id", 2);

        const std::size_t hd = std::size_t(config.head_dim);
        // LLaMA 3.1's scaling, as llama.cpp's converter stores it: a factor
        // per frequency that divides it.
        if (config.rope_scaling) {
            std::vector<float> factors(hd / 2);
            for (std::size_t i = 0; i < hd / 2; ++i) {
                const double inv =
                    std::pow(double(config.rope_theta), -2.0 * double(i) / double(hd));
                factors[i] = float(inv / llama3_inv_freq(inv, *config.rope_scaling));
            }
            g.tensor("rope_freqs.weight", {hd / 2}, vkml_test::kGgmlF32, vkml_test::raw(factors));
        }
        const auto permuted = [&](const std::vector<float>& w, std::size_t heads) {
            const std::size_t cols = w.size() / (heads * hd);
            std::vector<float> out(w.size());
            for (std::size_t h = 0; h < heads; ++h) {
                for (std::size_t half = 0; half < 2; ++half) {
                    for (std::size_t i = 0; i < hd / 2; ++i) {
                        const std::size_t from = h * hd + half * hd / 2 + i;
                        const std::size_t to = h * hd + 2 * i + half;
                        std::copy_n(w.begin() + std::ptrdiff_t(from * cols), cols,
                                    out.begin() + std::ptrdiff_t(to * cols));
                    }
                }
            }
            return out;
        };
        const auto put = [&](const std::string& name, const std::vector<float>& values,
                             const vkml::Shape& shape) {
            std::vector<std::uint64_t> dims(shape.rbegin(), shape.rend());
            g.tensor(name, dims, vkml_test::kGgmlF32, vkml_test::raw(values));
        };
        put("token_embd.weight", w("model.embed_tokens.weight"),
            shapes.at("model.embed_tokens.weight"));
        // Gemma's converter adds the 1 to its norm weights.
        const auto shifted = [&](std::vector<float> values) {
            for (float& v : values) v += 1.0f;
            return values;
        };
        put("output_norm.weight", gemma ? shifted(w("model.norm.weight")) : w("model.norm.weight"),
            shapes.at("model.norm.weight"));
        if (!config.tie_word_embeddings) {
            put("output.weight", w("lm_head.weight"), shapes.at("lm_head.weight"));
        }
        std::vector<std::pair<std::string, std::string>> names{
            {"input_layernorm.weight", "attn_norm.weight"},
            {"self_attn.q_proj.weight", "attn_q.weight"},
            {"self_attn.k_proj.weight", "attn_k.weight"},
            {"self_attn.v_proj.weight", "attn_v.weight"},
            {"self_attn.o_proj.weight", "attn_output.weight"},
            {"self_attn.q_proj.bias", "attn_q.bias"},
            {"self_attn.k_proj.bias", "attn_k.bias"},
            {"self_attn.v_proj.bias", "attn_v.bias"},
            {"self_attn.q_norm.weight", "attn_q_norm.weight"},
            {"self_attn.k_norm.weight", "attn_k_norm.weight"},
            {"post_attention_layernorm.weight", "ffn_norm.weight"},
            {"mlp.gate_proj.weight", "ffn_gate.weight"},
            {"mlp.up_proj.weight", "ffn_up.weight"},
            {"mlp.down_proj.weight", "ffn_down.weight"}};
        if (gemma) {
            // Its ffn_norm is the MLP's input norm; attention's output has its own.
            std::erase_if(names, [](const auto& n) { return n.second == "ffn_norm.weight"; });
            names.insert(names.end(),
                         {{"post_attention_layernorm.weight", "post_attention_norm.weight"},
                          {"pre_feedforward_layernorm.weight", "ffn_norm.weight"},
                          {"post_feedforward_layernorm.weight", "post_ffw_norm.weight"}});
        }
        for (int l = 0; l < config.num_layers; ++l) {
            const std::string hf = "model.layers." + std::to_string(l) + ".";
            const std::string gg = "blk." + std::to_string(l) + ".";
            for (const auto& [from, to] : names) {
                if (!weights.contains(hf + from)) continue;
                std::vector<float> values = w(hf + from);
                // SmolLM3's converter is LLaMA's, reordering q and k too.
                const bool reorder = arch == Arch::llama || arch == Arch::smollm3;
                if (reorder && from == "self_attn.q_proj.weight") {
                    values = permuted(values, std::size_t(config.num_heads));
                } else if (reorder && from == "self_attn.k_proj.weight") {
                    values = permuted(values, std::size_t(config.num_kv_heads));
                } else if (gemma && from.ends_with("norm.weight")) {
                    values = shifted(values);
                }
                put(gg + to, values, shapes.at(hf + from));
            }
        }
        const auto path = vkml_test::temp_path(file);
        g.write(path);
        return path;
    }

    const std::vector<float>& w(const std::string& name) const { return weights.at(name); }

    // Logits for the last of tokens, run from scratch in double precision.
    std::vector<double> reference_logits(const std::vector<std::int32_t>& tokens) const {
        const std::size_t t_len = tokens.size(), d = std::size_t(config.hidden_size);
        const std::size_t heads = std::size_t(config.num_heads), hd = std::size_t(config.head_dim);
        const std::size_t kv_heads = std::size_t(config.num_kv_heads);
        using Rows = std::vector<std::vector<double>>;

        const auto matvec = [](const std::vector<float>& m, const std::vector<double>& x,
                               std::size_t out) {
            std::vector<double> y(out, 0.0);
            for (std::size_t o = 0; o < out; ++o) {
                for (std::size_t i = 0; i < x.size(); ++i)
                    y[o] += double(m[o * x.size() + i]) * x[i];
            }
            return y;
        };
        const bool gemma = arch == Arch::gemma2 || arch == Arch::gemma3;
        const auto rms_norm = [&](const std::vector<double>& x, const std::vector<float>& weight) {
            double ss = 0.0;
            for (double v : x) ss += v * v;
            const double scale =
                1.0 / std::sqrt(ss / double(x.size()) + double(config.rms_norm_eps));
            std::vector<double> y(x.size());
            for (std::size_t i = 0; i < x.size(); ++i)
                y[i] = x[i] * scale * ((gemma ? 1.0 : 0.0) + double(weight[i]));
            return y;
        };
        // Gemma's sliding layers rotate at their own base, without scaling.
        const auto rope = [&](std::vector<double>& v, std::size_t n_heads, std::size_t pos,
                              bool local) {
            const double theta =
                local ? double(*config.sliding_rope_theta) : double(config.rope_theta);
            for (std::size_t h = 0; h < n_heads; ++h) {
                for (std::size_t i = 0; i < hd / 2; ++i) {
                    double inv = std::pow(theta, -2.0 * double(i) / double(hd));
                    if (config.rope_scaling && !local)
                        inv = llama3_inv_freq(inv, *config.rope_scaling);
                    if (!local) inv /= double(config.rope_linear_factor);
                    const double angle = double(pos) * inv;
                    double& a = v[h * hd + i];
                    double& b = v[h * hd + i + hd / 2];  // rotate-half pairs, as HF uses
                    const double a0 = a, b0 = b;
                    a = a0 * std::cos(angle) - b0 * std::sin(angle);
                    b = a0 * std::sin(angle) + b0 * std::cos(angle);
                }
            }
        };

        Rows x(t_len);
        for (std::size_t t = 0; t < t_len; ++t) {
            const auto& e = w("model.embed_tokens.weight");
            x[t].assign(e.begin() + std::ptrdiff_t(std::size_t(tokens[t]) * d),
                        e.begin() + std::ptrdiff_t((std::size_t(tokens[t]) + 1) * d));
            if (gemma) {
                for (double& value : x[t]) value *= std::sqrt(double(d));
            }
            if (arch == Arch::granite) {
                for (double& value : x[t]) value *= double(config.embedding_scale);
            }
        }
        for (int l = 0; l < config.num_layers; ++l) {
            const std::string p = "model.layers." + std::to_string(l) + ".";
            const bool slides = config.sliding_window && (config.sliding_layers.empty() ||
                                                          config.sliding_layers[std::size_t(l)]);
            Rows q(t_len), k(t_len), v(t_len);
            for (std::size_t t = 0; t < t_len; ++t) {
                const auto h =
                    arch == Arch::olmo2 ? x[t] : rms_norm(x[t], w(p + "input_layernorm.weight"));
                q[t] = matvec(w(p + "self_attn.q_proj.weight"), h, heads * hd);
                k[t] = matvec(w(p + "self_attn.k_proj.weight"), h, kv_heads * hd);
                v[t] = matvec(w(p + "self_attn.v_proj.weight"), h, kv_heads * hd);
                if (arch == Arch::qwen2) {
                    for (const auto& [out, name] :
                         {std::pair{&q[t], "q"}, {&k[t], "k"}, {&v[t], "v"}}) {
                        const auto& bias = w(p + "self_attn." + name + "_proj.bias");
                        for (std::size_t i = 0; i < out->size(); ++i) (*out)[i] += double(bias[i]);
                    }
                }
                if (arch == Arch::qwen3 || arch == Arch::gemma3) {
                    // Each head of q and k normalized on its own, sharing one weight.
                    for (const auto& [out, name] : {std::pair{&q[t], "q"}, {&k[t], "k"}}) {
                        const auto& weight = w(p + "self_attn." + name + "_norm.weight");
                        std::vector<double> one(hd);
                        for (std::size_t head = 0; head * hd < out->size(); ++head) {
                            for (std::size_t i = 0; i < hd; ++i) one[i] = (*out)[head * hd + i];
                            const auto normed = rms_norm(one, weight);
                            for (std::size_t i = 0; i < hd; ++i) (*out)[head * hd + i] = normed[i];
                        }
                    }
                }
                if (arch == Arch::olmo2) {
                    q[t] = rms_norm(q[t], w(p + "self_attn.q_norm.weight"));
                    k[t] = rms_norm(k[t], w(p + "self_attn.k_norm.weight"));
                }
                if (config.rope_layers.empty() || config.rope_layers[std::size_t(l)]) {
                    rope(q[t], heads, t, config.sliding_rope_theta && slides);
                    rope(k[t], kv_heads, t, config.sliding_rope_theta && slides);
                }
            }
            for (std::size_t t = 0; t < t_len; ++t) {
                std::vector<double> attn(heads * hd, 0.0);
                for (std::size_t h = 0; h < heads; ++h) {
                    const std::size_t kh = h / (heads / kv_heads);
                    // Keys first..t: all of them, or the last window.
                    const std::size_t window = slides ? std::size_t(*config.sliding_window) : t + 1;
                    const std::size_t first = t + 1 > window ? t + 1 - window : 0;
                    std::vector<double> s(t + 1, -1e300);
                    double max = -1e300, sum = 0.0;
                    for (std::size_t j = first; j <= t; ++j) {
                        s[j] = 0.0;
                        for (std::size_t c = 0; c < hd; ++c)
                            s[j] += q[t][h * hd + c] * k[j][kh * hd + c];
                        // Gemma: over sqrt(query_pre_attn_scalar).
                        s[j] /= std::sqrt(double(config.query_pre_attn_scalar.value_or(float(hd))));
                        // Gemma 2: capped.
                        if (const double cap = config.attn_logit_softcap; cap > 0)
                            s[j] = cap * std::tanh(s[j] / cap);
                        max = std::max(max, s[j]);
                    }
                    for (double& e : s) sum += (e = std::exp(e - max));
                    for (std::size_t j = first; j <= t; ++j) {
                        for (std::size_t c = 0; c < hd; ++c) {
                            attn[h * hd + c] += s[j] / sum * v[j][kh * hd + c];
                        }
                    }
                }
                auto o = matvec(w(p + "self_attn.o_proj.weight"), attn, d);
                const bool post = gemma || arch == Arch::olmo2;
                if (post) o = rms_norm(o, w(p + "post_attention_layernorm.weight"));
                for (std::size_t i = 0; i < d; ++i) x[t][i] += o[i] * config.residual_scale;

                const auto h2 =
                    arch == Arch::olmo2
                        ? x[t]
                        : rms_norm(x[t], w(p + (gemma ? "pre_feedforward_layernorm.weight"
                                                      : "post_attention_layernorm.weight")));
                auto gate = matvec(w(p + "mlp.gate_proj.weight"), h2,
                                   std::size_t(config.intermediate_size));
                const auto up =
                    matvec(w(p + "mlp.up_proj.weight"), h2, std::size_t(config.intermediate_size));
                for (std::size_t i = 0; i < gate.size(); ++i) {
                    const double g = gate[i];
                    const double act =
                        config.activation == vkml::Activation::gelu_tanh
                            ? 0.5 * g *
                                  (1.0 + std::tanh(0.7978845608028654 * (g + 0.044715 * g * g * g)))
                            : g / (1.0 + std::exp(-g));
                    gate[i] = act * up[i];
                }
                auto down = matvec(w(p + "mlp.down_proj.weight"), gate, d);
                if (post) down = rms_norm(down, w(p + "post_feedforward_layernorm.weight"));
                for (std::size_t i = 0; i < d; ++i) x[t][i] += down[i] * config.residual_scale;
            }
        }
        const auto last = rms_norm(x.back(), w("model.norm.weight"));
        const auto& head =
            config.tie_word_embeddings ? w("model.embed_tokens.weight") : w("lm_head.weight");
        auto logits = matvec(head, last, std::size_t(config.vocab_size));
        for (double& l : logits) l /= double(config.logit_divisor);
        if (const double cap = config.final_logit_softcap; cap > 0) {
            for (double& l : logits) l = cap * std::tanh(l / cap);
        }
        return logits;
    }
};

std::size_t count_mismatches(const std::vector<float>& got, const std::vector<double>& want) {
    std::size_t bad = got.size() == want.size() ? 0 : 1;
    for (std::size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
        if (!(std::abs(got[i] - want[i]) <= 1e-4 * (1.0 + std::abs(want[i])))) ++bad;
    }
    return bad;
}

const std::vector<std::int32_t> kPrompt{1, 5, 36, 0, 17, 17, 2, 30, 11};

}  // namespace

TEST_CASE("LlamaConfig reads an HF config.json and fills in defaults", "[llama]") {
    const auto path = vkml_test::temp_path("vkml_llama_config.json");
    std::ofstream(path) << R"({"vocab_size": 32000, "hidden_size": 4096,
        "intermediate_size": 11008, "num_hidden_layers": 32, "num_attention_heads": 32,
        "max_position_embeddings": 4096, "rms_norm_eps": 1e-06, "eos_token_id": 2})";
    const LlamaConfig c = LlamaConfig::from_json(path);
    CHECK(c.vocab_size == 32000);
    CHECK(c.num_layers == 32);
    CHECK(c.num_kv_heads == 32);  // absent: plain multi-head attention
    CHECK(c.head_dim == 128);     // absent: hidden_size / num_attention_heads
    CHECK(c.rms_norm_eps == 1e-6f);
    CHECK(c.rope_theta == 10000.0f);
    CHECK_FALSE(c.tie_word_embeddings);
    CHECK(c.eos_token_ids == std::vector<std::int32_t>{2});
}

TEST_CASE("LlamaConfig reads Qwen2 and LLaMA biases and rejects what it cannot run", "[llama]") {
    const auto config = [](const std::string& name, const std::string& extra) {
        const auto path = vkml_test::temp_path(name);
        std::ofstream(path) << R"({"vocab_size": 8, "hidden_size": 8, "intermediate_size": 8,
            "num_hidden_layers": 1, "num_attention_heads": 2, "max_position_embeddings": 8)"
                            << extra << "}";
        return path;
    };
    const LlamaConfig llama = LlamaConfig::from_json(config("vkml_cfg_llama.json", ""));
    CHECK_FALSE(llama.qkv_bias);
    CHECK_FALSE(llama.o_bias);
    const LlamaConfig qwen2 = LlamaConfig::from_json(
        config("vkml_cfg_qwen2.json", R"(, "model_type": "qwen2", "use_sliding_window": false)"));
    CHECK(qwen2.qkv_bias);
    CHECK_FALSE(qwen2.o_bias);
    const LlamaConfig biased = LlamaConfig::from_json(
        config("vkml_cfg_bias.json", R"(, "model_type": "llama", "attention_bias": true)"));
    CHECK(biased.qkv_bias);
    CHECK(biased.o_bias);
    CHECK_FALSE(qwen2.qk_norm);
    const LlamaConfig qwen3 = LlamaConfig::from_json(
        config("vkml_cfg_qwen3.json",
               R"(, "model_type": "qwen3", "attention_bias": false, "use_sliding_window": false,
           "sliding_window": null, "head_dim": 128)"));
    CHECK(qwen3.qk_norm);
    CHECK_FALSE(qwen3.qkv_bias);
    CHECK_FALSE(qwen3.o_bias);
    CHECK(qwen3.head_dim == 128);
    CHECK_FALSE(qwen3.sliding_window.has_value());

    // Sliding windows are read, with the layers they apply to.
    CHECK_FALSE(qwen2.sliding_window.has_value());
    const LlamaConfig qwen2_swa = LlamaConfig::from_json(
        config("vkml_cfg_swa.json", R"(, "model_type": "qwen2", "use_sliding_window": true,
                                       "sliding_window": 4096, "max_window_layers": 1)"));
    CHECK(qwen2_swa.sliding_window == 4096);
    CHECK(qwen2_swa.sliding_layers == std::vector<bool>{false});  // from layer 1 of 1
    const LlamaConfig typed = LlamaConfig::from_json(
        config("vkml_cfg_layer_types.json", R"(, "model_type": "qwen2", "use_sliding_window": true,
            "sliding_window": 64, "layer_types": ["sliding_attention"])"));
    CHECK(typed.sliding_layers == std::vector<bool>{true});
    REQUIRE_THROWS_WITH(LlamaConfig::from_json(config("vkml_cfg_layer_types_bad.json",
                                                      R"(, "layer_types": ["linear_attention"])")),
                        ContainsSubstring("linear_attention"));
    const LlamaConfig mistral = LlamaConfig::from_json(
        config("vkml_cfg_mistral.json", R"(, "model_type": "mistral", "sliding_window": 4096)"));
    CHECK(mistral.sliding_window == 4096);
    CHECK_FALSE(mistral.qkv_bias);
    const LlamaConfig mistral3 = LlamaConfig::from_json(
        config("vkml_cfg_mistral3.json", R"(, "model_type": "mistral", "sliding_window": null)"));
    CHECK_FALSE(mistral3.sliding_window.has_value());

    // GELU's tanh approximation is vkml's gelu; exact GELU (erf) is not.
    CHECK(LlamaConfig::from_json(
              config("vkml_cfg_gelu_tanh.json", R"(, "hidden_act": "gelu_pytorch_tanh")"))
              .activation == vkml::Activation::gelu_tanh);
    CHECK(LlamaConfig::from_json(config("vkml_cfg_gelu_new.json", R"(, "hidden_act": "gelu_new")"))
              .activation == vkml::Activation::gelu_tanh);
    CHECK(llama.activation == vkml::Activation::silu);
    REQUIRE_THROWS_WITH(
        LlamaConfig::from_json(config("vkml_cfg_gelu.json", R"(, "hidden_act": "gelu")")),
        ContainsSubstring("gelu"));
    REQUIRE_THROWS_WITH(
        LlamaConfig::from_json(config("vkml_cfg_mlp_bias.json", R"(, "mlp_bias": true)")),
        ContainsSubstring("mlp_bias"));
    REQUIRE_THROWS_WITH(
        LlamaConfig::from_json(config("vkml_cfg_gemma.json", R"(, "model_type": "gemma")")),
        ContainsSubstring("gemma"));
}

TEST_CASE("LlamaConfig reads rope settings as transformers 5 writes them", "[llama]") {
    const auto config = [](const std::string& name, const std::string& rope) {
        const auto path = vkml_test::temp_path(name);
        std::ofstream(path) << R"({"vocab_size": 8, "hidden_size": 8, "intermediate_size": 8,
            "num_hidden_layers": 1, "num_attention_heads": 2, "max_position_embeddings": 8,
            "rope_parameters": )"
                            << rope << "}";
        return path;
    };
    const LlamaConfig plain = LlamaConfig::from_json(
        config("vkml_rope_params.json", R"({"rope_theta": 1000000.0, "rope_type": "default"})"));
    CHECK(plain.rope_theta == 1000000.0f);
    CHECK_FALSE(plain.rope_scaling.has_value());

    const LlamaConfig scaled = LlamaConfig::from_json(
        config("vkml_rope_params31.json",
               R"({"rope_theta": 500000.0, "rope_type": "llama3", "factor": 8.0,
            "low_freq_factor": 1.0, "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192})"));
    CHECK(scaled.rope_theta == 500000.0f);
    REQUIRE(scaled.rope_scaling.has_value());
    CHECK(scaled.rope_scaling->factor == 8.0f);
    CHECK(scaled.rope_scaling->original_max_positions == 8192);

    CHECK(LlamaConfig::from_json(
              config("vkml_rope_params_linear.json", R"({"rope_type": "linear", "factor": 8.0})"))
              .rope_linear_factor == 8.0f);
    REQUIRE_THROWS_WITH(LlamaConfig::from_json(config("vkml_rope_params_yarn.json",
                                                      R"({"rope_type": "yarn", "factor": 4.0})")),
                        ContainsSubstring("yarn"));
}

TEST_CASE("Llama with a sliding window matches the reference", "[llama]") {
    TinyModel model{false};
    // A window shorter than the prompt, in every layer or only the second.
    model.config.sliding_window = 3;
    const bool one_layer = GENERATE(false, true);
    CAPTURE(one_layer);
    if (one_layer) model.config.sliding_layers = {false, true};
    vkml::Context context;
    Llama llama = Llama::load(context, model.write("vkml_tiny_swa"), 32);
    // All at once, then again a token at a time through the KV cache.
    const std::vector<double> want = model.reference_logits(kPrompt);
    CHECK(count_mismatches(llama.forward(kPrompt).to_vector<float>(), want) == 0);
    llama.reset();
    std::vector<float> last;
    for (const std::int32_t token : kPrompt) last = llama.forward({&token, 1}).to_vector<float>();
    CHECK(count_mismatches(last, want) == 0);
    CHECK(context.validation_error_count() == 0);
}

// 23 tokens: long enough for a window of 3 in a ring of 5 rows to wrap
// around several times.
const std::vector<std::int32_t> kLongPrompt{1,  5, 36, 0,  17, 17, 2,  30, 11, 4, 9, 22,
                                            13, 8, 8,  35, 20, 3,  27, 6,  14, 1, 33};

TEST_CASE("Sliding layers keep only the window and a few rows more", "[llama]") {
    // A window of 3 in every layer (Mistral), or in the first of two with a
    // rope base of its own (Gemma 3) or a soft cap on the scores (Gemma 2).
    const Arch arch = GENERATE(Arch::llama, Arch::gemma3, Arch::gemma2);
    CAPTURE(int(arch));
    TinyModel model{true, false, false, arch};
    if (arch == Arch::llama) model.config.sliding_window = 3;
    vkml::Context context;
    const auto dir = model.write("vkml_tiny_swa_ring_" + std::to_string(int(arch)));
    // Rings of 3 + 2 rows: a prompt goes in 3 queries at a time once it wraps.
    vkml::LlamaOptions ring, plenty;
    ring.sliding_extra_rows = 2;
    plenty.sliding_extra_rows = 100;
    Llama llama = Llama::load(context, dir, 32, ring);
    const Llama whole = Llama::load(context, dir, 32, plenty);
    const std::int64_t row_bytes = 2 * model.config.num_kv_heads * model.config.head_dim * 4;
    const std::int64_t sliding = arch == Arch::llama ? 2 : 1;
    CHECK(whole.kv_cache_bytes() == 2 * 32 * row_bytes);
    CHECK(llama.kv_cache_bytes() == ((2 - sliding) * 32 + sliding * 5) * row_bytes);

    const std::vector<double> want = model.reference_logits(kLongPrompt);
    CHECK(count_mismatches(llama.forward(kLongPrompt).to_vector<float>(), want) == 0);
    // A token at a time, checked all the way.
    llama.reset();
    for (std::size_t n = 1; n <= kLongPrompt.size(); ++n) {
        CAPTURE(n);
        const std::vector<std::int32_t> prefix(kLongPrompt.begin(),
                                               kLongPrompt.begin() + std::ptrdiff_t(n));
        CHECK(count_mismatches(llama.forward({&kLongPrompt[n - 1], 1}).to_vector<float>(),
                               model.reference_logits(prefix)) == 0);
    }
    // In steps of every size.
    llama.reset();
    std::vector<float> last;
    std::size_t at = 0;
    for (const std::size_t n : std::vector<std::size_t>{7, 1, 4, 2, 1, 8}) {
        last = llama.forward(std::span{kLongPrompt}.subspan(at, n)).to_vector<float>();
        at += n;
    }
    REQUIRE(at == kLongPrompt.size());
    CHECK(count_mismatches(last, want) == 0);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Llama rewinds sliding layers only as far as their rings still reach", "[llama]") {
    TinyModel model{false};
    model.config.sliding_window = 3;
    vkml::Context context;
    vkml::LlamaOptions ring;
    ring.sliding_extra_rows = 2;
    Llama llama = Llama::load(context, model.write("vkml_tiny_swa_rewind"), 32, ring);
    const std::vector<double> want = model.reference_logits(kLongPrompt);
    const std::span<const std::int32_t> prompt{kLongPrompt};

    // 20 tokens, then 2 wrong ones: the rings of 5 hold positions 17 to 21,
    // and position 20's queries need keys from 18 on, so it can be returned to.
    (void)llama.forward(prompt.first(20));
    (void)llama.forward(std::vector<std::int32_t>{3, 3});
    CHECK(llama.rewind(20) == 20);
    CHECK(llama.position() == 20);
    CHECK(count_mismatches(llama.forward(prompt.subspan(20)).to_vector<float>(), want) == 0);

    // Position 10's keys are long gone: the sequence starts over instead.
    CHECK(llama.rewind(10) == 0);
    CHECK(llama.position() == 0);
    CHECK(count_mismatches(llama.forward(prompt).to_vector<float>(), want) == 0);
    REQUIRE_THROWS_WITH(llama.rewind(llama.position() + 1), ContainsSubstring("rewind"));
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("LlamaConfig reads a list of end-of-sequence tokens", "[llama]") {
    const auto path = vkml_test::temp_path("vkml_llama_eos.json");
    std::ofstream(path) << R"({"vocab_size": 8, "hidden_size": 8, "intermediate_size": 8,
        "num_hidden_layers": 1, "num_attention_heads": 2, "max_position_embeddings": 8,
        "eos_token_id": [5, 7]})";
    CHECK(LlamaConfig::from_json(path).eos_token_ids == std::vector<std::int32_t>{5, 7});
}

TEST_CASE("LlamaConfig reads llama3 rope scaling and rejects other kinds", "[llama]") {
    const auto config = [](const std::string& name, const std::string& scaling) {
        const auto path = vkml_test::temp_path(name);
        std::ofstream(path) << R"({"vocab_size": 8, "hidden_size": 8, "intermediate_size": 8,
            "num_hidden_layers": 1, "num_attention_heads": 2, "max_position_embeddings": 8,
            "rope_scaling": )"
                            << scaling << "}";
        return path;
    };
    const LlamaConfig c =
        LlamaConfig::from_json(config("vkml_llama31.json", R"({"rope_type": "llama3",
        "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192})"));
    REQUIRE(c.rope_scaling.has_value());
    CHECK(c.rope_scaling->factor == 8.0f);
    CHECK(c.rope_scaling->low_freq_factor == 1.0f);
    CHECK(c.rope_scaling->high_freq_factor == 4.0f);
    CHECK(c.rope_scaling->original_max_positions == 8192);

    REQUIRE_THROWS_WITH(LlamaConfig::from_json(config("vkml_llama_yarn.json",
                                                      R"({"rope_type": "yarn", "factor": 4.0})")),
                        ContainsSubstring("yarn"));
}

TEST_CASE("Llama prefill matches a double-precision reference", "[llama]") {
    const bool tied = GENERATE(false, true);
    const bool bf16 = GENERATE(false, true);
    CAPTURE(tied, bf16);
    const TinyModel model{tied, bf16};
    vkml::Context context;
    Llama llama = Llama::load(context, model.write("vkml_tiny_llama"), 32);

    const vkml::Tensor logits = llama.forward(kPrompt);
    CHECK(logits.shape() == vkml::Shape{model.config.vocab_size});
    CHECK(count_mismatches(logits.to_vector<float>(), model.reference_logits(kPrompt)) == 0);
    CHECK(llama.position() == std::int64_t(kPrompt.size()));
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Qwen2 with q, k and v biases matches the reference", "[llama]") {
    const TinyModel model{true, false, false, Arch::qwen2};
    vkml::Context context;
    Llama llama = Llama::load(context, model.write("vkml_tiny_qwen2"), 32);
    REQUIRE(llama.config().qkv_bias);
    // All at once, then again a token at a time through the KV cache.
    CHECK(count_mismatches(llama.forward(kPrompt).to_vector<float>(),
                           model.reference_logits(kPrompt)) == 0);
    llama.reset();
    std::vector<float> last;
    for (const std::int32_t token : kPrompt) last = llama.forward({&token, 1}).to_vector<float>();
    CHECK(count_mismatches(last, model.reference_logits(kPrompt)) == 0);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Qwen3 with q and k normalized per head matches the reference", "[llama]") {
    const TinyModel model{true, false, false, Arch::qwen3};
    const bool gguf = GENERATE(false, true);
    CAPTURE(gguf);
    vkml::Context context;
    Llama llama = Llama::load(
        context, gguf ? model.write_gguf("vkml_tiny_qwen3.gguf") : model.write("vkml_tiny_qwen3"),
        32);
    const LlamaConfig& c = llama.config();
    REQUIRE(c.qk_norm);
    CHECK_FALSE(c.qkv_bias);
    CHECK(c.head_dim == 16);
    CHECK(c.rope_style == vkml::RopeStyle::RotateHalf);
    // All at once, then again a token at a time through the KV cache.
    CHECK(count_mismatches(llama.forward(kPrompt).to_vector<float>(),
                           model.reference_logits(kPrompt)) == 0);
    llama.reset();
    std::vector<float> last;
    for (const std::int32_t token : kPrompt) last = llama.forward({&token, 1}).to_vector<float>();
    CHECK(count_mismatches(last, model.reference_logits(kPrompt)) == 0);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Gemma 3 matches the reference", "[llama]") {
    const TinyModel model{true, false, false, Arch::gemma3};
    vkml::Context context;
    Llama llama = Llama::load(context, model.write("vkml_tiny_gemma3"), 32);
    const LlamaConfig& c = llama.config();
    REQUIRE(c.sandwich_norms);
    CHECK(c.norm_weight_offset == 1.0f);
    CHECK(c.sliding_layers == std::vector<bool>{true, false});
    CHECK(c.sliding_rope_theta == 100.0f);
    CHECK(c.query_pre_attn_scalar == 8.0f);
    CHECK(c.activation == vkml::Activation::gelu_tanh);
    CHECK(c.tie_word_embeddings);
    // All at once, then again a token at a time through the KV cache.
    const std::vector<double> want = model.reference_logits(kPrompt);
    CHECK(count_mismatches(llama.forward(kPrompt).to_vector<float>(), want) == 0);
    llama.reset();
    std::vector<float> last;
    for (const std::int32_t token : kPrompt) last = llama.forward({&token, 1}).to_vector<float>();
    CHECK(count_mismatches(last, want) == 0);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Gemma 2 matches the reference", "[llama]") {
    const TinyModel model{true, false, false, Arch::gemma2};
    vkml::Context context;
    Llama llama = Llama::load(context, model.write("vkml_tiny_gemma2"), 32);
    const LlamaConfig& c = llama.config();
    REQUIRE(c.sandwich_norms);
    CHECK_FALSE(c.qk_norm);
    CHECK(c.sliding_layers == std::vector<bool>{true, false});
    CHECK_FALSE(c.sliding_rope_theta);
    CHECK(c.attn_logit_softcap == 1.0f);
    CHECK(c.final_logit_softcap == 3.0f);
    // query_pre_attn_scalar 8 against a head_dim of 16, with no q norm to
    // fold it into. All at once, then a token at a time through the KV cache.
    const std::vector<double> want = model.reference_logits(kPrompt);
    CHECK(count_mismatches(llama.forward(kPrompt).to_vector<float>(), want) == 0);
    llama.reset();
    std::vector<float> last;
    for (const std::int32_t token : kPrompt) last = llama.forward({&token, 1}).to_vector<float>();
    CHECK(count_mismatches(last, want) == 0);
    for (const double l : want) CHECK(std::abs(l) < 3.0);  // the cap holds
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Gemma 2 from a GGUF file matches the reference", "[llama]") {
    // As llama.cpp reads it: the caps from the file, and scores over
    // sqrt(head_dim) but for 27B.
    TinyModel model{true, false, false, Arch::gemma2};
    model.config.query_pre_attn_scalar.reset();
    vkml::Context context;
    Llama llama = Llama::load(context, model.write_gguf("vkml_tiny_gemma2.gguf"), 32);
    CHECK(llama.config().sliding_layers == std::vector<bool>{true, false});
    CHECK(llama.config().attn_logit_softcap == 1.0f);
    CHECK(llama.config().final_logit_softcap == 3.0f);
    const std::vector<double> want = model.reference_logits(kPrompt);
    CHECK(count_mismatches(llama.forward(kPrompt).to_vector<float>(), want) == 0);
    llama.reset();
    std::vector<float> last;
    for (const std::int32_t token : kPrompt) last = llama.forward({&token, 1}).to_vector<float>();
    CHECK(count_mismatches(last, want) == 0);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Gemma 3's image-text checkpoints run their text model", "[llama]") {
    TinyModel model{true, false, false, Arch::gemma3};
    model.image_text = true;
    vkml::Context context;
    Llama llama = Llama::load(context, model.write("vkml_tiny_gemma3_image_text"), 32);
    CHECK(llama.config().sandwich_norms);
    CHECK(llama.config().eos_token_ids == std::vector<std::int32_t>{2, 7});
    CHECK(count_mismatches(llama.forward(kPrompt).to_vector<float>(),
                           model.reference_logits(kPrompt)) == 0);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Gemma 3's configs leave out what they keep at its defaults", "[llama]") {
    // Gemma 3 4B's text_config, which names only what differs.
    const auto path = vkml_test::temp_path("vkml_cfg_gemma3_4b.json");
    std::ofstream(path) << R"({"architectures": ["Gemma3ForConditionalGeneration"],
        "model_type": "gemma3", "eos_token_id": [1, 106],
        "text_config": {"hidden_size": 2560, "intermediate_size": 10240,
            "model_type": "gemma3_text", "num_attention_heads": 8, "num_hidden_layers": 34,
            "num_key_value_heads": 4, "rope_scaling": {"factor": 8.0, "rope_type": "linear"},
            "sliding_window": 1024}})";
    const LlamaConfig c = LlamaConfig::from_json(path);
    CHECK(c.vocab_size == 262208);
    CHECK(c.hidden_size == 2560);
    CHECK(c.num_layers == 34);
    CHECK(c.head_dim == 256);
    CHECK(c.max_positions == 131072);
    CHECK(c.rms_norm_eps == 1e-6f);
    CHECK(c.rope_theta == 1000000.0f);
    CHECK(c.rope_linear_factor == 8.0f);
    CHECK(c.sliding_rope_theta == 10000.0f);
    CHECK(c.sliding_window == 1024);
    REQUIRE(c.sliding_layers.size() == 34);
    CHECK_FALSE(c.sliding_layers[5]);  // every sixth layer is global
    CHECK(c.sliding_layers[6]);
    CHECK(c.query_pre_attn_scalar == 256.0f);
    CHECK(c.activation == vkml::Activation::gelu_tanh);
    CHECK(c.tie_word_embeddings);
    CHECK(c.eos_token_ids == std::vector<std::int32_t>{1, 106});
}

TEST_CASE("Gemma 3 with linear rope scaling matches the reference", "[llama]") {
    // As 4B and up: positions in the global layer divided by 8, the sliding
    // one's unscaled; from a config and from a GGUF file.
    TinyModel model{true, false, false, Arch::gemma3};
    model.config.rope_linear_factor = 8.0f;
    const bool gguf = GENERATE(false, true);
    CAPTURE(gguf);
    if (gguf) {
        // As llama.cpp reads it (see below): both layers slide, so the
        // scaling it reads must change neither.
        model.config.sliding_layers = {true, true};
        model.config.sliding_rope_theta = 10000.0f;
        model.config.query_pre_attn_scalar.reset();
    }
    vkml::Context context;
    Llama llama = Llama::load(context,
                              gguf ? model.write_gguf("vkml_tiny_gemma3_linear.gguf")
                                   : model.write("vkml_tiny_gemma3_linear"),
                              32);
    CHECK(llama.config().rope_linear_factor == 8.0f);
    CHECK(count_mismatches(llama.forward(kPrompt).to_vector<float>(),
                           model.reference_logits(kPrompt)) == 0);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Gemma 3 from a GGUF file matches the reference", "[llama]") {
    // As llama.cpp reads it: every layer of two slides (a global one every
    // sixth), at rope base 10000, and scores over sqrt(head_dim).
    TinyModel model{true, false, false, Arch::gemma3};
    model.config.sliding_layers = {true, true};
    model.config.sliding_rope_theta = 10000.0f;
    model.config.query_pre_attn_scalar.reset();
    vkml::Context context;
    Llama llama = Llama::load(context, model.write_gguf("vkml_tiny_gemma3.gguf"), 32);
    CHECK(llama.config().sandwich_norms);
    CHECK(llama.config().norm_weight_offset == 0.0f);  // the file's weights have the 1
    const std::vector<double> want = model.reference_logits(kPrompt);
    CHECK(count_mismatches(llama.forward(kPrompt).to_vector<float>(), want) == 0);
    llama.reset();
    std::vector<float> last;
    for (const std::int32_t token : kPrompt) last = llama.forward({&token, 1}).to_vector<float>();
    CHECK(count_mismatches(last, want) == 0);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Llama with the tanh GELU activation matches the reference", "[llama]") {
    TinyModel model{false};
    model.config.activation = vkml::Activation::gelu_tanh;
    vkml::Context context;
    Llama llama = Llama::load(context, model.write("vkml_tiny_llama_gelu"), 32);
    REQUIRE(llama.config().activation == vkml::Activation::gelu_tanh);
    CHECK(count_mismatches(llama.forward(kPrompt).to_vector<float>(),
                           model.reference_logits(kPrompt)) == 0);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Llama with an f16 KV cache stays close to the reference", "[llama]") {
    const TinyModel model{false};
    vkml::Context context;
    vkml::LlamaOptions options;
    options.kv_cache = vkml::DType::F16;
    Llama llama = Llama::load(context, model.write("vkml_tiny_llama_f16_cache"), 32, options);
    const std::vector<double> want = model.reference_logits(kPrompt);
    const auto max_error = [&](const std::vector<float>& got) {
        double worst = 0.0;
        for (std::size_t i = 0; i < want.size(); ++i) {
            worst = std::max(worst, std::abs(got[i] - want[i]) / (1.0 + std::abs(want[i])));
        }
        return worst;
    };
    // Keys and values rounded to f16 (11 significant bits) move the logits by
    // about 5e-4 of their size; unrounded, the tests above hold them to 1e-4.
    const double prefill = max_error(llama.forward(kPrompt).to_vector<float>());
    llama.reset();
    std::vector<float> last;
    for (const std::int32_t token : kPrompt) last = llama.forward({&token, 1}).to_vector<float>();
    const double decode = max_error(last);
    CAPTURE(prefill, decode);
    CHECK(prefill > 0.0);  // the cache did round
    CHECK(prefill < 5e-3);
    CHECK(decode < 5e-3);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Llama loads a GGUF file as llama.cpp's converter writes it", "[llama]") {
    const bool tied = GENERATE(false, true);
    const bool scaled = GENERATE(false, true);  // LLaMA 3.1's rope scaling, as rope_freqs
    CAPTURE(tied, scaled);
    const TinyModel model{tied, false, scaled};
    vkml::Context context;
    Llama llama = Llama::load(context, model.write_gguf("vkml_tiny_llama.gguf"), 32);
    const LlamaConfig& c = llama.config();
    CHECK(c.num_layers == model.config.num_layers);
    CHECK(c.num_kv_heads == model.config.num_kv_heads);
    CHECK(c.vocab_size == model.config.vocab_size);
    CHECK(c.tie_word_embeddings == tied);
    CHECK(c.rope_style == vkml::RopeStyle::Interleaved);
    CHECK(c.eos_token_ids == std::vector<std::int32_t>{2});
    CHECK(c.rope_freq_factors.size() == (scaled ? std::size_t(c.head_dim / 2) : 0));
    // Its permuted q and k rows under interleaved rope give HF's logits.
    CHECK(count_mismatches(llama.forward(kPrompt).to_vector<float>(),
                           model.reference_logits(kPrompt)) == 0);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Llama with quantized weights stays close to the reference", "[llama]") {
    const TinyModel model{false};
    vkml::Context context;
    const vkml::QuantType type =
        GENERATE(vkml::QuantType::q8_0, vkml::QuantType::q4_0, vkml::QuantType::q4_1);
    const bool q4 = type != vkml::QuantType::q8_0;
    CAPTURE(q4);
    // Width 32 matrices quantize; down_proj (48 columns) cannot and stays as is.
    Llama llama = Llama::load(context, model.write("vkml_tiny_llama_quantized"), 32,
                              vkml::LlamaOptions{.quantize = type});
    const std::vector<float> got = llama.forward(kPrompt).to_vector<float>();
    const std::vector<double> want = model.reference_logits(kPrompt);

    double max_logit = 0.0, max_diff = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        max_logit = std::max(max_logit, std::abs(want[i]));
        max_diff = std::max(max_diff, std::abs(got[i] - want[i]));
    }
    CAPTURE(max_diff, max_logit);
    CHECK(max_diff > 0.0);  // it did quantize
    // About 2% of the range with 8 bits per weight (and per activation, on
    // GPUs with integer dot products); steps 16 times coarser with 4 bits
    // give about 20% on this tiny random model, whose 32-wide rows are one
    // block each. (The kernels are checked exactly against dequantized
    // weights and activations in test_quantize.cpp.)
    CHECK(max_diff < (q4 ? 0.3 : 0.04) * max_logit);
    const auto argmax = [](const auto& v) {
        return std::max_element(v.begin(), v.end()) - v.begin();
    };
    CHECK(argmax(got) == argmax(want));
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Llama with llama3 rope scaling matches the reference", "[llama]") {
    const TinyModel model{false, false, true};
    vkml::Context context;
    Llama llama = Llama::load(context, model.write("vkml_tiny_llama_scaled"), 32);
    CHECK(count_mismatches(llama.forward(kPrompt).to_vector<float>(),
                           model.reference_logits(kPrompt)) == 0);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Llama decoding through the KV cache matches running the whole prefix", "[llama]") {
    const TinyModel model{false};
    vkml::Context context;
    Llama llama = Llama::load(context, model.write("vkml_tiny_llama_decode"), 32);

    // A 4-token prompt, then one token at a time.
    const std::vector<std::int32_t> prompt(kPrompt.begin(), kPrompt.begin() + 4);
    CHECK(count_mismatches(llama.forward(prompt).to_vector<float>(),
                           model.reference_logits(prompt)) == 0);
    for (std::size_t n = 5; n <= kPrompt.size(); ++n) {
        const std::vector<std::int32_t> prefix(kPrompt.begin(),
                                               kPrompt.begin() + std::ptrdiff_t(n));
        const std::vector<float> got = llama.forward({&kPrompt[n - 1], 1}).to_vector<float>();
        CHECK(count_mismatches(got, model.reference_logits(prefix)) == 0);
    }
    CHECK(llama.position() == std::int64_t(kPrompt.size()));

    llama.reset();  // a new sequence reuses the cache from position 0
    CHECK(count_mismatches(llama.forward(prompt).to_vector<float>(),
                           model.reference_logits(prompt)) == 0);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("OLMo 2 with output norms and whole-projection q and k norms matches the reference",
          "[llama]") {
    const TinyModel model{false, false, false, Arch::olmo2};
    vkml::Context context;
    Llama llama = Llama::load(context, model.write("vkml_tiny_olmo2"), 32);
    CHECK_FALSE(llama.config().pre_norms);
    CHECK(llama.config().qk_norm_whole);
    // All at once, then again a token at a time through the KV cache.
    CHECK(count_mismatches(llama.forward(kPrompt).to_vector<float>(),
                           model.reference_logits(kPrompt)) == 0);
    llama.reset();
    std::vector<float> last;
    for (const std::int32_t token : kPrompt) last = llama.forward({&token, 1}).to_vector<float>();
    CHECK(count_mismatches(last, model.reference_logits(kPrompt)) == 0);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("SmolLM3 with no rope in some layers matches the reference", "[llama]") {
    const TinyModel model{true, false, false, Arch::smollm3};
    // GGUF files leave the layers out: llama.cpp fixes every fourth.
    const bool gguf = GENERATE(false, true);
    CAPTURE(gguf);
    vkml::Context context;
    Llama llama = Llama::load(
        context,
        gguf ? model.write_gguf("vkml_tiny_smollm3.gguf") : model.write("vkml_tiny_smollm3"), 32);
    CHECK(llama.config().rope_layers == std::vector<bool>{true, true, true, false});
    // All at once, then again a token at a time through the KV cache.
    CHECK(count_mismatches(llama.forward(kPrompt).to_vector<float>(),
                           model.reference_logits(kPrompt)) == 0);
    llama.reset();
    std::vector<float> last;
    for (const std::int32_t token : kPrompt) last = llama.forward({&token, 1}).to_vector<float>();
    CHECK(count_mismatches(last, model.reference_logits(kPrompt)) == 0);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Granite with its four multipliers matches the reference", "[llama]") {
    const TinyModel model{true, false, false, Arch::granite};
    vkml::Context context;
    Llama llama = Llama::load(context, model.write("vkml_tiny_granite"), 32);
    CHECK(llama.config().embedding_scale == 3.0f);
    CHECK(llama.config().query_pre_attn_scalar == 16.0f);
    CHECK(llama.config().residual_scale == 0.5f);
    CHECK(llama.config().logit_divisor == 2.0f);
    // All at once, then again a token at a time through the KV cache.
    CHECK(count_mismatches(llama.forward(kPrompt).to_vector<float>(),
                           model.reference_logits(kPrompt)) == 0);
    llama.reset();
    std::vector<float> last;
    for (const std::int32_t token : kPrompt) last = llama.forward({&token, 1}).to_vector<float>();
    CHECK(count_mismatches(last, model.reference_logits(kPrompt)) == 0);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Llama can rewind and continue a sequence differently", "[llama]") {
    const TinyModel model{false};
    vkml::Context context;
    Llama llama = Llama::load(context, model.write("vkml_tiny_llama_rewind"), 32);

    // Run a prefix, then a wrong continuation; rewind to the prefix and take
    // the right one: the result must be as if the wrong one never happened.
    const std::vector<std::int32_t> prefix(kPrompt.begin(), kPrompt.begin() + 5);
    (void)llama.forward(prefix);
    (void)llama.forward(std::vector<std::int32_t>{3, 3, 3});
    llama.rewind(5);
    CHECK(llama.position() == 5);
    const std::vector<std::int32_t> rest(kPrompt.begin() + 5, kPrompt.end());
    CHECK(count_mismatches(llama.forward(rest).to_vector<float>(),
                           model.reference_logits(kPrompt)) == 0);

    REQUIRE_THROWS_WITH(llama.rewind(llama.position() + 1), ContainsSubstring("rewind"));
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("Llama rejects sequences beyond its context and checkpoints missing weights", "[llama]") {
    const TinyModel model{false};
    vkml::Context context;
    const auto dir = model.write("vkml_tiny_llama_errors");
    Llama llama = Llama::load(context, dir, 8);
    REQUIRE_THROWS_WITH(llama.forward(kPrompt), ContainsSubstring("8"));
    REQUIRE_THROWS_WITH(llama.forward({}), ContainsSubstring("no tokens"));

    std::filesystem::remove(dir / "model-00002-of-00002.safetensors");
    REQUIRE_THROWS_WITH(Llama::load(context, dir, 8), ContainsSubstring("model.layers.1."));
}
