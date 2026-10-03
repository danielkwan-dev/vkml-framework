// Times the matmuls a LLaMA forward pass spends its time in: a weight matrix
// [n, k] applied to m rows of activations, as matmul_transposed. m = 1 is one
// decoding step, which reads every weight once per token and so is bound by
// memory bandwidth; larger m is prefill, which is bound by arithmetic.
//
//   vkml-bench [--no-dot] [--device <name substring>] [--decode <model> [--f32-scales]]
//
// --no-dot times quantized weights without integer dot products (see
// ContextOptions::integer_dot_product).
//
// --decode times instead the matrix-vector products of one decoding step of
// a real model (an HF directory or a GGUF file; only its config is read):
// every matrix of a layer and the output projection, in q8_0, q4_0 and q4_1,
// each many times. GPU timings vary by up to half between identical runs, so
// it reports the minimum, which is steady, next to the median, and the
// minimums summed over a token: the weights' share of a decoding step.
// --f32-scales gives its matrices f32 scales instead of f16, to compare.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <vkml/llama.hpp>
#include <vkml/vkml.hpp>

namespace {

using Clock = std::chrono::steady_clock;

// How the weights are stored: bf16 as checkpoints ship them, or quantized.
enum class Weights { f32, bf16, q8, q4 };

struct Case {
    const char* name;
    std::int64_t m, k, n;
    Weights weights = Weights::f32;
};

using enum Weights;

// TinyLlama 1.1B shapes; LLaMA 7B's are the same with every size doubled or so.
constexpr Case kCases[] = {
    {"tiny", 1, 16, 16},
    {"decode q_proj", 1, 2048, 2048},
    {"decode gate/up", 1, 2048, 5632},
    {"decode down", 1, 5632, 2048},
    {"decode lm_head", 1, 2048, 32000},
    {"m=4 gate/up", 4, 2048, 5632},
    {"m=8 gate/up", 8, 2048, 5632},
    {"m=16 gate/up", 16, 2048, 5632},
    {"m=32 gate/up", 32, 2048, 5632},
    {"m=64 gate/up", 64, 2048, 5632},
    {"prefill q_proj", 128, 2048, 2048},
    {"prefill gate/up", 128, 2048, 5632},
    {"bf16 q_proj", 1, 2048, 2048, bf16},
    {"bf16 k/v_proj", 1, 2048, 256, bf16},
    {"bf16 gate/up", 1, 2048, 5632, bf16},
    {"bf16 down", 1, 5632, 2048, bf16},
    {"bf16 lm_head", 1, 2048, 32000, bf16},
    {"bf16 m=2 gate/up", 2, 2048, 5632, bf16},
    {"bf16 m=3 gate/up", 3, 2048, 5632, bf16},
    {"bf16 m=4 gate/up", 4, 2048, 5632, bf16},
    {"bf16 m=6 gate/up", 6, 2048, 5632, bf16},
    {"bf16 m=8 gate/up", 8, 2048, 5632, bf16},
    {"bf16 prefill g/u", 128, 2048, 5632, bf16},
    {"q8 q_proj", 1, 2048, 2048, q8},
    {"q8 gate/up", 1, 2048, 5632, q8},
    {"q8 down", 1, 5632, 2048, q8},
    {"q8 lm_head", 1, 2048, 32000, q8},
    {"q8 prefill g/u", 128, 2048, 5632, q8},
    {"q4 q_proj", 1, 2048, 2048, q4},
    {"q4 gate/up", 1, 2048, 5632, q4},
    {"q4 down", 1, 5632, 2048, q4},
    {"q4 lm_head", 1, 2048, 32000, q4},
    {"q4 prefill g/u", 128, 2048, 5632, q4},
};

// Bytes per weight: quantized formats add an f16 scale per block of 32.
double bytes_per_weight(Weights w) {
    switch (w) {
        case f32: return 4.0;
        case bf16: return 2.0;
        case q8: return 1.0 + 2.0 / 32;
        case q4: return 0.5 + 2.0 / 32;
    }
    return 0.0;
}

// The best and the median time per run of fn over batches of runs
// back-to-back: batches amortize the submit and readback, and the best batch
// discounts GPU clocks still ramping up.
struct Timing {
    double best = 0.0, median = 0.0;  // seconds
};

Timing time_runs(const std::function<vkml::Tensor(int)>& fn, int batches, int runs) {
    std::vector<double> per_run;
    for (int batch = 0; batch < batches; ++batch) {
        const auto start = Clock::now();
        std::vector<vkml::Tensor> outs;
        for (int i = 0; i < runs; ++i) outs.push_back(fn(i));
        (void)outs.back().to_bytes();
        per_run.push_back(std::chrono::duration<double>(Clock::now() - start).count() / runs);
    }
    std::ranges::sort(per_run);
    return {per_run.front(), per_run[per_run.size() / 2]};
}

// A matrix [rows, cols] of type whose every weight is exactly 0.25, built
// from its bytes: no f32 copy, which for Gemma's 262144-row output
// projection would exceed a GPU buffer. q8_0: q = 64 and d = 1 / 256;
// q4_0: q = 4, stored as 12, and d = 1 / 16; q4_1: q = 1, d = 0.25, min 0.
// Scales in f16 (whose bits these powers of two have exactly), or f32.
vkml::QuantizedMatrix quarters(vkml::Context& context, vkml::QuantType type, std::int64_t rows,
                               std::int64_t cols, bool f32_scales) {
    using vkml::QuantType;
    const bool q8 = type == QuantType::q8_0;
    const bool offsets = type == QuantType::q4_1;
    const std::uint32_t word = q8 ? 0x40404040u : offsets ? 0x11111111u : 0xCCCCCCCCu;
    const std::int64_t words = cols / (q8 ? 4 : 8);
    const std::vector<std::uint32_t> values(std::size_t(rows * words), word);
    std::vector<float> scales(std::size_t(rows * cols / 32 * (offsets ? 2 : 1)));
    for (std::size_t i = 0; i < scales.size(); ++i) {
        scales[i] = offsets ? (i % 2 == 0 ? 0.25f : 0.0f) : q8 ? 1.0f / 256 : 1.0f / 16;
    }
    const vkml::Shape scale_shape =
        offsets ? vkml::Shape{rows, cols / 32, 2} : vkml::Shape{rows, cols / 32};
    std::vector<std::uint16_t> halves;
    for (const float x : scales) {
        halves.push_back(x == 0.25f        ? 0x3400
                         : x == 1.0f / 256 ? 0x1C00
                         : x == 1.0f / 16  ? 0x2C00
                                           : 0);
    }
    return {vkml::Tensor::from_bytes(context, std::as_bytes(std::span{values}), {rows, words},
                                     vkml::DType::I32),
            f32_scales ? vkml::Tensor::from_data<float>(context, scales, scale_shape)
                       : vkml::Tensor::from_bytes(context, std::as_bytes(std::span{halves}),
                                                  scale_shape, vkml::DType::F16),
            rows, cols, type};
}

// See --decode above.
int bench_decode(vkml::Context& context, const std::filesystem::path& model, bool f32_scales) {
    using vkml::QuantType;
    const vkml::LlamaConfig c = model.extension() == ".gguf"
                                    ? vkml::LlamaConfig::from_gguf(model)
                                    : vkml::LlamaConfig::from_json(model / "config.json");
    const std::int64_t d = c.hidden_size, f = c.intermediate_size, layers = c.num_layers;
    const std::int64_t q_dim = c.num_heads * c.head_dim, kv_dim = c.num_kv_heads * c.head_dim;
    struct Matrix {
        const char* name;
        std::int64_t rows, cols, per_token;
    };
    const double scale_bytes = f32_scales ? 4.0 : 2.0;
    const auto bytes = [&](QuantType type, const auto& m) {
        const double per_weight = type == QuantType::q8_0   ? 1.0 + scale_bytes / 32
                                  : type == QuantType::q4_0 ? 0.5 + scale_bytes / 32
                                                            : 0.5 + 2 * scale_bytes / 32;
        return double(m.rows * m.cols) * per_weight;
    };
    const Matrix matrices[] = {
        {"q_proj", q_dim, d, layers}, {"k/v_proj", kv_dim, d, 2 * layers},
        {"o_proj", d, q_dim, layers}, {"gate/up", f, d, 2 * layers},
        {"down", d, f, layers},       {"lm_head", c.vocab_size, d, 1},
    };
    // The GPU's clocks shift for tens of seconds at a time, moving even the
    // best of a format's runs by a quarter: rounds that each time every
    // format, taking each matrix's best over them all, see several such
    // spells.
    constexpr int kRounds = 4;
    const std::pair<QuantType, const char*> formats[] = {
        {QuantType::q8_0, "q8_0"}, {QuantType::q4_0, "q4_0"}, {QuantType::q4_1, "q4_1"}};
    constexpr std::size_t kMatrices = std::size(matrices);
    std::vector<std::vector<std::vector<Timing>>> timings(
        std::size(formats), std::vector<std::vector<Timing>>(kMatrices));
    float value = 0.0f;
    for (int round = 0; round < kRounds; ++round) {
        for (std::size_t fi = 0; fi < std::size(formats); ++fi) {
            const QuantType type = formats[fi].first;
            for (std::size_t mi = 0; mi < kMatrices; ++mi) {
                const Matrix& m = matrices[mi];
                if (m.cols % 32 != 0) continue;  // stays unquantized in a model
                value += 0.5f;  // as in the matmul cases: no recycled memory holds the answer
                const vkml::Tensor x = vkml::Tensor::from_data<float>(
                    context, std::vector<float>(std::size_t(m.cols), value), {1, m.cols});
                // Copies enough to overflow the GPU's caches, cycled through.
                const int copies = std::clamp(int(256e6 / bytes(type, m)), 1, 32);
                std::vector<vkml::QuantizedMatrix> weights;
                for (int i = 0; i < copies; ++i) {
                    weights.push_back(quarters(context, type, m.rows, m.cols, f32_scales));
                }
                const auto product = [&](int i) {
                    return vkml::matmul_transposed(x, weights[std::size_t(i % copies)]);
                };
                // Every output is value / 4 * cols, but for int8 activations' rounding.
                const float expected = value * 0.25f * float(m.cols);
                const std::vector<float> check = product(0).to_vector<float>();
                if (std::ranges::any_of(check, [&](float v) {
                        return !(std::abs(v - expected) <= expected * 1e-5f);
                    })) {
                    std::fprintf(stderr, "%s %s: wrong result, expected every output to be %g\n",
                                 formats[fi].second, m.name, double(expected));
                    return 1;
                }
                timings[fi][mi].push_back(time_runs(product, 10, 20));
            }
        }
    }

    std::printf("%-6s %-9s %7s %6s %8s %9s %9s %8s\n", "format", "matrix", "rows", "cols", "MB",
                "best ms", "median ms", "GB/s");
    for (std::size_t fi = 0; fi < std::size(formats); ++fi) {
        const auto [type, format] = formats[fi];
        double token_best = 0.0, token_median = 0.0, token_bytes = 0.0;
        for (std::size_t mi = 0; mi < kMatrices; ++mi) {
            const Matrix& m = matrices[mi];
            std::vector<Timing>& t = timings[fi][mi];
            if (t.empty()) continue;
            std::ranges::sort(t, {}, &Timing::median);
            const double median = t[t.size() / 2].median;
            const double best = std::ranges::min(t, {}, &Timing::best).best;
            std::printf("%-6s %-9s %7lld %6lld %8.1f %9.3f %9.3f %8.1f\n", format, m.name,
                        static_cast<long long>(m.rows), static_cast<long long>(m.cols),
                        bytes(type, m) / 1e6, best * 1e3, median * 1e3,
                        bytes(type, m) / best / 1e9);
            token_best += best * double(m.per_token);
            token_median += median * double(m.per_token);
            token_bytes += bytes(type, m) * double(m.per_token);
        }
        std::printf("%-6s %-9s %7s %6s %8.1f %9.2f %9.2f %8.1f\n\n", format, "per token", "", "",
                    token_bytes / 1e6, token_best * 1e3, token_median * 1e3,
                    token_bytes / token_best / 1e9);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    vkml::ContextOptions options;
    options.enable_validation = false;  // it would dominate the timings
    std::filesystem::path decode_model;
    bool f32_scales = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--device" && i + 1 < argc) {
            options.device_name = argv[++i];
        } else if (flag == "--decode" && i + 1 < argc) {
            decode_model = argv[++i];
        } else if (flag == "--f32-scales") {
            f32_scales = true;
        } else if (flag == "--no-dot") {
            options.integer_dot_product = false;
        } else {
            std::fprintf(stderr,
                         "usage: %s [--no-dot] [--device <name substring>] [--decode <model> "
                         "[--f32-scales]]\n",
                         argv[0]);
            return 2;
        }
    }

    try {
        vkml::Context context{options};
        std::printf("device  %s\n\n", context.device_info().name.c_str());
        if (!decode_model.empty()) return bench_decode(context, decode_model, f32_scales);
        std::printf("%-18s %6s %6s %6s %10s %10s %10s\n", "case", "m", "k", "n", "ms", "GB/s",
                    "GFLOP/s");

        float scale = 0.0f;
        for (const Case& c : kCases) {
            // A different value per case, so memory recycled from an earlier
            // case can never hold this case's answer.
            scale += 0.5f;
            const std::vector<float> a_host(static_cast<std::size_t>(c.m * c.k), scale);
            const std::vector<float> w_host(static_cast<std::size_t>(c.n * c.k), 0.25f);
            // 0x3E80 is 0.25 in bf16, so both kinds of case expect the same results.
            const std::vector<std::uint16_t> w_bf16(static_cast<std::size_t>(c.n * c.k), 0x3E80);
            const vkml::Tensor a = vkml::Tensor::from_data<float>(context, a_host, {c.m, c.k});
            // Distinct copies of the weights, cycled through, so no run finds
            // them in a cache: a model reads each weight once per token, and
            // integrated GPUs share a last-level cache big enough to hold a
            // whole small matrix, which would overstate bandwidth.
            const double weight_bytes = bytes_per_weight(c.weights) * double(c.n * c.k);
            const int copies = std::clamp(int(256e6 / weight_bytes), 1, 32);
            std::vector<vkml::Tensor> weights;
            std::vector<vkml::QuantizedMatrix> quantized;
            for (int i = 0; i < copies; ++i) {
                vkml::Tensor w =
                    c.weights == f32
                        ? vkml::Tensor::from_data<float>(context, w_host, {c.n, c.k})
                        : vkml::Tensor::from_bytes(context, std::as_bytes(std::span{w_bf16}),
                                                   {c.n, c.k}, vkml::DType::BF16);
                if (c.weights == q8) quantized.push_back(vkml::quantize_q8(w));
                if (c.weights == q4) quantized.push_back(vkml::quantize_q4(w));
                if (quantized.empty()) weights.push_back(std::move(w));
            }
            const auto product = [&](int i) {
                const std::size_t at = std::size_t(i % copies);
                return quantized.empty() ? vkml::matmul_transposed(a, weights[at])
                                         : vkml::matmul_transposed(a, quantized[at]);
            };
            // Warm up (pipeline creation, caches) and check the result: every
            // output is scale / 4 * k, so a kernel that skips work cannot pass
            // for a fast one. Exactly, but for q8_0, whose scale 0.25 / 127
            // rounds to f16 (by up to 2^-11).
            const std::vector<float> check = product(0).to_vector<float>();
            const float expected = scale * 0.25f * static_cast<float>(c.k);
            const float tolerance = c.weights == q8 ? expected * 1e-3f : 0.0f;
            if (std::ranges::any_of(check,
                                    [&](float v) { return std::abs(v - expected) > tolerance; })) {
                std::fprintf(stderr, "%s: wrong result, expected every output to be %g\n", c.name,
                             double(expected));
                return 1;
            }

            const double seconds = time_runs(product, 7, 20).best;

            const double bytes = 4.0 * double(c.m * c.k + c.m * c.n) + weight_bytes;
            const double flops = 2.0 * double(c.m) * double(c.k) * double(c.n);
            std::printf("%-18s %6lld %6lld %6lld %10.3f %10.1f %10.1f\n", c.name,
                        static_cast<long long>(c.m), static_cast<long long>(c.k),
                        static_cast<long long>(c.n), seconds * 1e3, bytes / seconds / 1e9,
                        flops / seconds / 1e9);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
