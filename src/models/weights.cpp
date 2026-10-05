#include <algorithm>
#include <string>

#include "io/gguf.hpp"
#include "models/internal.hpp"
#include "ops/internal.hpp"
#include "vkml/ops.hpp"

namespace vkml {

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
        if (const auto part = fused_part(g, shape)) {
            const auto& [whole, start] = *part;
            const Shape all{file_.shape(whole)[0], shape[1]};
            return rows_of(tensor_named(whole, all), start, shape[0]);
        }
        return tensor_named(g, shape);
    }

    std::optional<QuantizedMatrix> quantized(const std::string& name,
                                             const Shape& shape) const override {
        const std::string g = name_of(name);
        if (const auto part = fused_part(g, shape)) {
            const auto& [whole, start] = *part;
            auto q = quantized_named(whole, {file_.shape(whole)[0], shape[1]});
            if (!q) return std::nullopt;
            q->values = rows_of(q->values, start, shape[0]);
            q->scales = rows_of(q->scales, start, shape[0]);
            q->rows = shape[0];
            return q;
        }
        return quantized_named(g, shape);
    }

private:
    // Phi-3's fused projections, as llama.cpp keeps them: attn_qkv holds q's,
    // k's and v's rows, ffn_up gate's, then up's. For the part g of shape, the
    // whole and the row it starts at.
    std::optional<std::pair<std::string, std::int64_t>> fused_part(const std::string& g,
                                                                   const Shape& shape) const {
        if (shape.size() != 2) return std::nullopt;
        const auto in = [&](const std::string& part, const std::string& whole) {
            return g.ends_with(part) ? g.substr(0, g.size() - part.size()) + whole : std::string();
        };
        const std::int64_t rows = shape[0];
        if (const auto w = in("attn_q.weight", "attn_qkv.weight"); !w.empty() && file_.contains(w))
            return std::pair{w, std::int64_t{0}};
        if (const auto w = in("attn_k.weight", "attn_qkv.weight"); !w.empty() && file_.contains(w))
            return std::pair{w, file_.shape(w)[0] - 2 * rows};
        if (const auto w = in("attn_v.weight", "attn_qkv.weight"); !w.empty() && file_.contains(w))
            return std::pair{w, file_.shape(w)[0] - rows};
        if (const auto w = in("ffn_gate.weight", "ffn_up.weight");
            !w.empty() && !file_.contains(g) && file_.contains(w))
            return std::pair{w, std::int64_t{0}};
        if (g.ends_with("ffn_up.weight") && file_.contains(g) && file_.shape(g)[0] == 2 * rows &&
            !file_.contains(in("ffn_up.weight", "ffn_gate.weight")))
            return std::pair{g, rows};
        return std::nullopt;
    }

    // Rows start.. of t, a matrix or (quantized scales) more, by its first dimension.
    // f16 rows of odd width (scales of 32 columns) widen to f32, to copy whole words.
    static Tensor rows_of(const Tensor& t, std::int64_t start, std::int64_t rows) {
        const std::int64_t all = t.shape()[0], width = t.numel() / all;
        Shape shape = t.shape();
        shape[0] = rows;
        const Tensor words = t.dtype() == DType::F16 && width % 2 != 0 ? cast(t, DType::F32) : t;
        return detail::read_rows(words.reshape({1, all, width}), start, rows).reshape(shape);
    }

    Tensor tensor_named(const std::string& g, const Shape& shape) const {
        if (!file_.contains(g)) throw Error("Llama: the GGUF file has no tensor " + g);
        check_shape(g, file_.shape(g), shape);
        // An embedding table stored quantized widens to f16 for lookups.
        if (is_quantized(file_.type_name(g))) return file_.load_f16(context_, g);
        return as_loaded(file_.load(context_, g), shape);
    }

    std::optional<QuantizedMatrix> quantized_named(const std::string& g, const Shape& shape) const {
        if (!file_.contains(g)) return std::nullopt;
        const std::string type = file_.type_name(g);
        if (!is_quantized(type)) return std::nullopt;
        check_shape(g, file_.shape(g), shape);
        // vkml's own formats, and q4_K, whose sub-blocks
        // are Q4_1's.
        if (type == "q8_0" || type == "q4_0" || type == "q4_1" || type == "q4_K") {
            return file_.load_quantized(context_, g);
        }
        // The others (q5_0, q5_1, q2_K, q3_K, q5_K, q6_K), decoded on the host, run as
        // Q8_0: finer than any of them.
        return file_.load_q8(context_, g);
    }

    static bool is_quantized(const std::string& type) {
        return type != "f32" && type != "f16" && type != "bf16";
    }

private:
    Context& context_;
    const detail::Gguf& file_;
    bool sandwich_norms_;
};

// HF's safetensors shards.
class SafeTensorsWeights final : public detail::LlamaWeightSource {
public:
    SafeTensorsWeights(Context& context, std::span<const SafeTensors> shards)
        : context_(context), shards_(shards) {
        // An image-text model's language model (Gemma 3 4B and up) sits under
        // a prefix: language_model. up to transformers 4, model.language_model.
        // (and lm_head at the top) since.
        if (!has("model.embed_tokens.weight")) {
            if (has("language_model.model.embed_tokens.weight")) {
                model_ = "language_model.model.";
                head_ = "language_model.lm_head.";
            } else if (has("model.language_model.embed_tokens.weight")) {
                model_ = "model.language_model.";
            }
        }
    }

    bool contains(const std::string& name) const override { return has(stored(name)); }
    Tensor tensor(const std::string& name, const Shape& shape) const override {
        const std::string s = stored(name);
        if (!has(s) && shape.size() == 2) {
            if (const auto part = fused_part(s, shape[0])) {
                // Phi-3's fused projections: rows of the whole, as a
                // [1, rows, cols] tensor whose middle dimension read_rows takes.
                const auto& [whole, start] = *part;
                const Tensor w = tensor(whole.first, {whole.second, shape[1]});
                return detail::read_rows(w.reshape({1, whole.second, shape[1]}), start, shape[0])
                    .reshape(shape);
            }
        }
        for (const SafeTensors& shard : shards_) {
            if (!shard.contains(s)) continue;
            check_shape(s, shard.shape(s), shape);
            return as_loaded(shard.load(context_, s), shape);
        }
        throw Error("Llama: the checkpoint has no weight " + s);
    }
    // Floats, but a matrix too large for one GPU buffer (Gemma 2 2B's bf16
    // embeddings, 1.18 GB, where Iris Xe allows 1 GB) is quantized to Q8_0 on
    // the host as it is read.
    std::optional<QuantizedMatrix> quantized(const std::string& name,
                                             const Shape& shape) const override {
        const std::string s = stored(name);
        for (const SafeTensors& shard : shards_) {
            if (!shard.contains(s)) continue;
            check_shape(s, shard.shape(s), shape);
            if (shape.size() != 2 || shape[1] % 32 != 0) return std::nullopt;
            const std::uint64_t bytes =
                std::uint64_t(shape[0] * shape[1]) * (shard.dtype(s) == "F32" ? 4 : 2);
            if (bytes <= context_.device_info().max_storage_buffer_range) return std::nullopt;
            return shard.load_q8(context_, s);
        }
        return std::nullopt;
    }

private:
    // For q_proj, k_proj, v_proj, gate_proj or up_proj of rows rows, absent
    // as such: the fused weight that holds it, its rows, and where it starts.
    // q, k and v are qkv_proj's rows in that order (k and v of equal size),
    // gate and up gate_up_proj's halves.
    std::optional<std::pair<std::pair<std::string, std::int64_t>, std::int64_t>> fused_part(
        const std::string& s, std::int64_t rows) const {
        const auto replace = [&](const std::string& part, const std::string& whole) {
            return s.substr(0, s.size() - part.size()) + whole;
        };
        for (const auto& [part, whole] : std::initializer_list<std::pair<std::string, std::string>>{
                 {"self_attn.q_proj.weight", "self_attn.qkv_proj.weight"},
                 {"self_attn.k_proj.weight", "self_attn.qkv_proj.weight"},
                 {"self_attn.v_proj.weight", "self_attn.qkv_proj.weight"},
                 {"mlp.gate_proj.weight", "mlp.gate_up_proj.weight"},
                 {"mlp.up_proj.weight", "mlp.gate_up_proj.weight"}}) {
            if (!s.ends_with(part)) continue;
            const std::string fused = replace(part, whole);
            for (const SafeTensors& shard : shards_) {
                if (!shard.contains(fused)) continue;
                const std::int64_t total = shard.shape(fused).at(0);
                const std::int64_t start =
                    part.starts_with("self_attn.q") || part.starts_with("mlp.gate") ? 0
                    : part.starts_with("self_attn.k") ? total - 2 * rows
                                                      : total - rows;
                return std::pair{std::pair{fused, total}, start};
            }
        }
        return std::nullopt;
    }

    bool has(const std::string& stored_name) const {
        return std::ranges::any_of(shards_,
                                   [&](const SafeTensors& s) { return s.contains(stored_name); });
    }
    // HF's usual name for a weight, as this checkpoint stores it.
    std::string stored(const std::string& name) const {
        if (name.starts_with("model.")) return model_ + name.substr(6);
        if (name.starts_with("lm_head.")) return head_ + name.substr(8);
        return name;
    }

    Context& context_;
    std::span<const SafeTensors> shards_;
    std::string model_ = "model.", head_ = "lm_head.";
};

}  // namespace

std::unique_ptr<detail::LlamaWeightSource> detail::gguf_weights(Context& context, const Gguf& file,
                                                                bool sandwich_norms) {
    return std::make_unique<GgufWeights>(context, file, sandwich_norms);
}

std::unique_ptr<detail::LlamaWeightSource> detail::safetensors_weights(
    Context& context, std::span<const SafeTensors> shards) {
    return std::make_unique<SafeTensorsWeights>(context, shards);
}

}  // namespace vkml
