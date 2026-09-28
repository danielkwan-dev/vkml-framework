#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <vkml/llama.hpp>
#include <vkml/vkml.hpp>

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

// A tiny LLaMA with random weights: 2 layers, grouped-query attention with 2
// query heads per KV head, and an odd vocabulary size. As Qwen2, it has
// biases on the q, k and v projections.
struct TinyModel {
    LlamaConfig config;
    std::map<std::string, std::vector<float>> weights;
    std::map<std::string, vkml::Shape> shapes;

    bool bf16 = false;  // weights stored as bf16, as most HF checkpoints are

    bool qwen2 = false;

    explicit TinyModel(bool tie_embeddings, bool bf16_weights = false, bool rope_scaled = false,
                       bool qwen2_biases = false)
        : bf16(bf16_weights), qwen2(qwen2_biases) {
        config.vocab_size = 37;
        config.hidden_size = 32;
        config.intermediate_size = 48;
        config.num_layers = 2;
        config.num_heads = 4;
        config.num_kv_heads = 2;
        config.head_dim = 8;
        config.max_positions = 64;
        config.rms_norm_eps = 1e-5f;
        config.rope_theta = 10000.0f;
        config.tie_word_embeddings = tie_embeddings;
        // With head_dim 8 and theta 10000 the wavelengths are about 6, 63, 628
        // and 6283, so these settings keep one frequency, blend one and divide
        // two.
        if (rope_scaled) config.rope_scaling = vkml::RopeScaling{4.0f, 1.0f, 4.0f, 64};

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
            add(p + "input_layernorm.weight", {d}, 1.0f, 0.1f);
            add(p + "self_attn.q_proj.weight", {q_dim, d}, 0.0f, 0.2f);
            add(p + "self_attn.k_proj.weight", {kv_dim, d}, 0.0f, 0.2f);
            add(p + "self_attn.v_proj.weight", {kv_dim, d}, 0.0f, 0.2f);
            add(p + "self_attn.o_proj.weight", {d, q_dim}, 0.0f, 0.2f);
            if (qwen2) {
                add(p + "self_attn.q_proj.bias", {q_dim}, 0.0f, 0.5f);
                add(p + "self_attn.k_proj.bias", {kv_dim}, 0.0f, 0.5f);
                add(p + "self_attn.v_proj.bias", {kv_dim}, 0.0f, 0.5f);
            }
            add(p + "post_attention_layernorm.weight", {d}, 1.0f, 0.1f);
            add(p + "mlp.gate_proj.weight", {f, d}, 0.0f, 0.2f);
            add(p + "mlp.up_proj.weight", {f, d}, 0.0f, 0.2f);
            add(p + "mlp.down_proj.weight", {d, f}, 0.0f, 0.2f);
        }
        add("model.norm.weight", {d}, 1.0f, 0.1f);
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
        const auto dir = vkml_test::temp_path(dir_name);
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        std::ofstream(dir / "config.json")
            << (qwen2 ? R"({"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2",)"
                        R"( "use_sliding_window": false, "vocab_size": )"
                      : R"({"architectures": ["LlamaForCausalLM"], "vocab_size": )")
            << config.vocab_size << R"(, "hidden_size": )" << config.hidden_size
            << R"(, "intermediate_size": )" << config.intermediate_size
            << R"(, "num_hidden_layers": )" << config.num_layers << R"(, "num_attention_heads": )"
            << config.num_heads << R"(, "num_key_value_heads": )" << config.num_kv_heads
            << R"(, "max_position_embeddings": )" << config.max_positions
            << R"(, "rms_norm_eps": 1e-05, "rope_theta": 10000.0, "rope_scaling": )"
            << (config.rope_scaling
                    ? R"({"rope_type": "llama3", "factor": 4.0, "low_freq_factor": 1.0,)"
                      R"( "high_freq_factor": 4.0, "original_max_position_embeddings": 64})"
                    : "null")
            << ","
            << R"( "tie_word_embeddings": )" << (config.tie_word_embeddings ? "true" : "false")
            << "}";

        std::vector<vkml_test::Entry> shard1, shard2;
        for (const auto& [name, values] : weights) {
            auto& shard = name.find("layers.1.") != std::string::npos ? shard2 : shard1;
            if (bf16) {
                std::vector<std::uint16_t> halves(values.size());
                for (std::size_t i = 0; i < values.size(); ++i) {
                    std::uint32_t bits;
                    std::memcpy(&bits, &values[i], 4);
                    halves[i] = std::uint16_t(bits >> 16);
                }
                shard.push_back({name, "BF16", shapes.at(name), vkml_test::raw(halves)});
            } else {
                shard.push_back({name, "F32", shapes.at(name), vkml_test::raw(values)});
            }
        }
        vkml_test::write_safetensors(dir / "model-00001-of-00002.safetensors", shard1);
        vkml_test::write_safetensors(dir / "model-00002-of-00002.safetensors", shard2);
        return dir;
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
        const auto rms_norm = [&](const std::vector<double>& x, const std::vector<float>& weight) {
            double ss = 0.0;
            for (double v : x) ss += v * v;
            const double scale =
                1.0 / std::sqrt(ss / double(x.size()) + double(config.rms_norm_eps));
            std::vector<double> y(x.size());
            for (std::size_t i = 0; i < x.size(); ++i) y[i] = x[i] * scale * double(weight[i]);
            return y;
        };
        const auto rope = [&](std::vector<double>& v, std::size_t n_heads, std::size_t pos) {
            for (std::size_t h = 0; h < n_heads; ++h) {
                for (std::size_t i = 0; i < hd / 2; ++i) {
                    double inv = std::pow(double(config.rope_theta), -2.0 * double(i) / double(hd));
                    if (config.rope_scaling) inv = llama3_inv_freq(inv, *config.rope_scaling);
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
        }
        for (int l = 0; l < config.num_layers; ++l) {
            const std::string p = "model.layers." + std::to_string(l) + ".";
            Rows q(t_len), k(t_len), v(t_len);
            for (std::size_t t = 0; t < t_len; ++t) {
                const auto h = rms_norm(x[t], w(p + "input_layernorm.weight"));
                q[t] = matvec(w(p + "self_attn.q_proj.weight"), h, heads * hd);
                k[t] = matvec(w(p + "self_attn.k_proj.weight"), h, kv_heads * hd);
                v[t] = matvec(w(p + "self_attn.v_proj.weight"), h, kv_heads * hd);
                if (qwen2) {
                    for (const auto& [out, name] :
                         {std::pair{&q[t], "q"}, {&k[t], "k"}, {&v[t], "v"}}) {
                        const auto& bias = w(p + "self_attn." + name + "_proj.bias");
                        for (std::size_t i = 0; i < out->size(); ++i) (*out)[i] += double(bias[i]);
                    }
                }
                rope(q[t], heads, t);
                rope(k[t], kv_heads, t);
            }
            for (std::size_t t = 0; t < t_len; ++t) {
                std::vector<double> attn(heads * hd, 0.0);
                for (std::size_t h = 0; h < heads; ++h) {
                    const std::size_t kh = h / (heads / kv_heads);
                    std::vector<double> s(t + 1);
                    double max = -1e300, sum = 0.0;
                    for (std::size_t j = 0; j <= t; ++j) {
                        s[j] = 0.0;
                        for (std::size_t c = 0; c < hd; ++c)
                            s[j] += q[t][h * hd + c] * k[j][kh * hd + c];
                        s[j] /= std::sqrt(double(hd));
                        max = std::max(max, s[j]);
                    }
                    for (double& e : s) sum += (e = std::exp(e - max));
                    for (std::size_t j = 0; j <= t; ++j) {
                        for (std::size_t c = 0; c < hd; ++c) {
                            attn[h * hd + c] += s[j] / sum * v[j][kh * hd + c];
                        }
                    }
                }
                const auto o = matvec(w(p + "self_attn.o_proj.weight"), attn, d);
                for (std::size_t i = 0; i < d; ++i) x[t][i] += o[i];

                const auto h2 = rms_norm(x[t], w(p + "post_attention_layernorm.weight"));
                auto gate = matvec(w(p + "mlp.gate_proj.weight"), h2,
                                   std::size_t(config.intermediate_size));
                const auto up =
                    matvec(w(p + "mlp.up_proj.weight"), h2, std::size_t(config.intermediate_size));
                for (std::size_t i = 0; i < gate.size(); ++i) {
                    gate[i] = gate[i] / (1.0 + std::exp(-gate[i])) * up[i];
                }
                const auto down = matvec(w(p + "mlp.down_proj.weight"), gate, d);
                for (std::size_t i = 0; i < d; ++i) x[t][i] += down[i];
            }
        }
        const auto last = rms_norm(x.back(), w("model.norm.weight"));
        const auto& head =
            config.tie_word_embeddings ? w("model.embed_tokens.weight") : w("lm_head.weight");
        return matvec(head, last, std::size_t(config.vocab_size));
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

    REQUIRE_THROWS_WITH(LlamaConfig::from_json(config("vkml_cfg_swa.json",
                                                      R"(, "model_type": "qwen2",
                                                      "use_sliding_window": true)")),
                        ContainsSubstring("sliding"));
    REQUIRE_THROWS_WITH(
        LlamaConfig::from_json(config("vkml_cfg_mlp_bias.json", R"(, "mlp_bias": true)")),
        ContainsSubstring("mlp_bias"));
    REQUIRE_THROWS_WITH(
        LlamaConfig::from_json(config("vkml_cfg_gemma.json", R"(, "model_type": "gemma")")),
        ContainsSubstring("gemma"));
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
    const TinyModel model{true, false, false, true};
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

TEST_CASE("Llama with quantized weights stays close to the reference", "[llama]") {
    const TinyModel model{false};
    vkml::Context context;
    const vkml::QuantType type = GENERATE(vkml::QuantType::q8_0, vkml::QuantType::q4_0);
    const bool q4 = type == vkml::QuantType::q4_0;
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
    // About 1% of the range with 8 bits per weight; steps 16 times coarser
    // with 4 bits give about 20% on this tiny random model. (The kernels are
    // checked exactly against dequantized weights in test_quantize.cpp.)
    CHECK(max_diff < (q4 ? 0.3 : 0.02) * max_logit);
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
