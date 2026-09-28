// Runs a LLaMA-architecture model: the prompt in one forward pass, then greedy
// decoding one token at a time, with timings.
//
//   vkml-run --model <dir> --prompt "The capital of France is" [--generate 64]
//   vkml-run --model <dir> --tokens 1,450,7483 [--generate 16]
//
// Other options: [--context 512] [--top 5] [--ignore-eos] [--q8 | --q4]
// [--profile] [--perplexity] [--dump-logits <file>] [--device <name
// substring>]. --q8 and --q4 quantize the weights to 8 or 4 bits (Q8_0, Q4_0)
// on loading. --profile times every kernel on
// the GPU and prints where the prompt and the generation spent it.
// --perplexity scores the prompt instead of continuing it: the model reads it
// a token at a time, and the perplexity of each next token is printed, the
// usual measure of how well a model predicts text (lower is better).
//
// <dir> holds an HF checkpoint: config.json, *.safetensors and, for --prompt,
// tokenizer.json. With --prompt the continuation streams as text; generation
// stops at an end-of-sequence token unless --ignore-eos. --dump-logits writes
// the prompt's last-token logits as raw little-endian f32, which
// tools/compare_hf.py checks against HF transformers.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <vkml/vkml.hpp>

#include "text_stream.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

struct Args {
    std::string model;
    std::string prompt;
    std::vector<std::int32_t> tokens;
    bool ignore_eos = false;
    bool profile = false;
    std::optional<vkml::QuantType> quantize;
    bool perplexity = false;
    int generate = 16;
    std::int64_t context = 512;
    int top = 5;
    std::string dump_logits;
    std::string device;
};

std::vector<std::int32_t> parse_tokens(const std::string& text) {
    std::vector<std::int32_t> tokens;
    std::stringstream in(text);
    std::string item;
    while (std::getline(in, item, ',')) tokens.push_back(std::stoi(item));
    return tokens;
}

bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--ignore-eos") {
            args.ignore_eos = true;
            continue;
        }
        if (flag == "--q8" || flag == "--q4") {
            args.quantize = flag == "--q8" ? vkml::QuantType::q8_0 : vkml::QuantType::q4_0;
            continue;
        }
        if (flag == "--perplexity") {
            args.perplexity = true;
            continue;
        }
        if (flag == "--profile") {
            args.profile = true;
            continue;
        }
        if (i + 1 == argc) return false;  // every other flag takes a value
        const std::string value = argv[++i];
        if (flag == "--model") {
            args.model = value;
        } else if (flag == "--prompt") {
            args.prompt = value;
        } else if (flag == "--tokens") {
            args.tokens = parse_tokens(value);
        } else if (flag == "--generate") {
            args.generate = std::stoi(value);
        } else if (flag == "--context") {
            args.context = std::stoll(value);
        } else if (flag == "--top") {
            args.top = std::stoi(value);
        } else if (flag == "--dump-logits") {
            args.dump_logits = value;
        } else if (flag == "--device") {
            args.device = value;
        } else {
            return false;
        }
    }
    return !args.model.empty() && (args.tokens.empty() != args.prompt.empty());
}

// GPU time per kernel, slowest first, as totals and per step.
void print_profile(const char* title, const std::vector<vkml::KernelTime>& profile, double steps,
                   double wall_seconds) {
    double total = 0.0;
    for (const auto& k : profile) total += k.milliseconds;
    std::printf("\n%s: GPU busy %.2f ms per step, wall clock %.2f ms per step\n", title,
                total / steps, wall_seconds * 1e3 / steps);
    std::printf("  %-12s %8s %12s %12s %7s\n", "kernel", "calls", "total ms", "ms/step", "share");
    for (const auto& k : profile) {
        std::printf("  %-12s %8llu %12.2f %12.3f %6.1f%%\n", k.name.c_str(),
                    static_cast<unsigned long long>(k.calls), k.milliseconds,
                    k.milliseconds / steps, 100.0 * k.milliseconds / total);
    }
}

std::int32_t argmax(const std::vector<float>& v) {
    return static_cast<std::int32_t>(std::max_element(v.begin(), v.end()) - v.begin());
}

void print_top(const std::vector<float>& logits, int top) {
    std::vector<std::int32_t> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    const auto k = std::min<std::size_t>(static_cast<std::size_t>(top), order.size());
    std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(k), order.end(),
                      [&](std::int32_t a, std::int32_t b) {
                          return logits[std::size_t(a)] > logits[std::size_t(b)];
                      });
    for (std::size_t i = 0; i < k; ++i) {
        std::printf("  %6d  %9.4f\n", order[i], double(logits[std::size_t(order[i])]));
    }
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    try {
        if (!parse_args(argc, argv, args)) {
            std::fprintf(stderr,
                         "usage: %s --model <dir> (--prompt <text> | --tokens <id,id,...>) "
                         "[--generate N] [--context N] [--top N] [--ignore-eos] "
                         "[--q8 | --q4] [--profile] [--perplexity] [--dump-logits <file>] "
                         "[--device <name>]\n",
                         argv[0]);
            return 2;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bad argument: %s\n", e.what());
        return 2;
    }

    try {
        vkml::ContextOptions options;
        options.device_name = args.device;
        vkml::Context context{options};
        std::printf("device   %s\n", context.device_info().name.c_str());

        auto start = Clock::now();
        vkml::Llama model = vkml::Llama::load(context, args.model, args.context,
                                              vkml::LlamaOptions{.quantize = args.quantize});
        const vkml::LlamaConfig& c = model.config();
        std::printf(
            "model    %lld layers, hidden %lld, %lld/%lld heads, vocab %lld (%.1f s to load)\n",
            static_cast<long long>(c.num_layers), static_cast<long long>(c.hidden_size),
            static_cast<long long>(c.num_heads), static_cast<long long>(c.num_kv_heads),
            static_cast<long long>(c.vocab_size), seconds_since(start));

        std::optional<vkml::Tokenizer> tokenizer;
        if (!args.prompt.empty()) {
            tokenizer.emplace(std::filesystem::path(args.model) / "tokenizer.json");
            args.tokens = tokenizer->encode(args.prompt);
            std::printf("prompt   %zu tokens\n", args.tokens.size());
        }

        if (args.perplexity) {
            if (args.tokens.size() < 2) throw vkml::Error("--perplexity needs at least 2 tokens");
            start = Clock::now();
            std::vector<float> logits = model.forward({&args.tokens[0], 1}).to_vector<float>();
            double nll = 0.0;
            for (std::size_t i = 1; i < args.tokens.size(); ++i) {
                // -log softmax(logits)[next], computed stably in double.
                const double max = *std::max_element(logits.begin(), logits.end());
                double sum = 0.0;
                for (const float l : logits) sum += std::exp(double(l) - max);
                nll += max + std::log(sum) - double(logits[std::size_t(args.tokens[i])]);
                if (i + 1 < args.tokens.size()) {
                    logits = model.forward({&args.tokens[i], 1}).to_vector<float>();
                }
            }
            const double n = double(args.tokens.size() - 1);
            std::printf(
                "perplexity %.4f over %zu tokens (mean negative log-likelihood %.5f, %.1f s)\n",
                std::exp(nll / n), args.tokens.size() - 1, nll / n, seconds_since(start));
            return 0;
        }

        if (args.profile) context.set_profiling(true);
        start = Clock::now();
        std::vector<float> logits = model.forward(args.tokens).to_vector<float>();
        const double prefill = seconds_since(start);
        if (args.profile) {
            print_profile("prompt", context.profile(), 1.0, prefill);
            context.reset_profile();
        }
        std::printf("prefill  %zu tokens in %.3f s (%.1f tokens/s)\n", args.tokens.size(), prefill,
                    double(args.tokens.size()) / prefill);
        std::printf("top logits after the prompt:\n");
        print_top(logits, args.top);

        if (!args.dump_logits.empty()) {
            std::ofstream out(args.dump_logits, std::ios::binary);
            out.write(reinterpret_cast<const char*>(logits.data()),
                      static_cast<std::streamsize>(logits.size() * sizeof(float)));
        }

        const auto& eos = c.eos_token_ids;
        std::vector<std::int32_t> generated;
        // Text streams as the difference between decodings of everything so far:
        // decoding generated tokens alone would drop the first one's space.
        std::vector<std::int32_t> all = args.tokens;
        std::size_t printed = 0;
        if (tokenizer) {
            const std::string prompt_text = tokenizer->decode(all);
            std::printf("\n%s", prompt_text.c_str());
            printed = prompt_text.size();
        }
        start = Clock::now();
        for (int i = 0; i < args.generate && model.position() < model.context_length(); ++i) {
            const std::int32_t next = argmax(logits);
            if (!args.ignore_eos && std::ranges::find(eos, next) != eos.end()) break;
            generated.push_back(next);
            all.push_back(next);
            if (tokenizer) {
                const std::string text = tokenizer->decode(all);
                const std::size_t ready = complete_utf8_prefix(text);
                if (ready > printed) {
                    std::printf("%s", text.substr(printed, ready - printed).c_str());
                    std::fflush(stdout);
                    printed = ready;
                }
            }
            logits = model.forward({&next, 1}).to_vector<float>();
        }
        if (tokenizer) std::printf("\n\n");
        if (!generated.empty()) {
            const double decode = seconds_since(start);
            if (args.profile) {
                print_profile("generation", context.profile(), double(generated.size()), decode);
            }
            std::printf("decode   %zu tokens in %.3f s (%.1f tokens/s)\n", generated.size(), decode,
                        double(generated.size()) / decode);
            if (!tokenizer) {
                std::printf("generated");
                for (std::size_t i = 0; i < generated.size(); ++i) {
                    std::printf("%s%d", i ? "," : " ", generated[i]);
                }
                std::printf("\n");
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
