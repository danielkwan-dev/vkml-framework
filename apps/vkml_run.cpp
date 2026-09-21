// Runs a LLaMA-architecture model on token ids: the prompt in one forward
// pass, then greedy decoding one token at a time, with timings.
//
//   vkml-run --model <dir> --tokens 1,450,7483 [--generate 16] [--context 512]
//            [--top 5] [--dump-logits <file>] [--device <name substring>]
//
// <dir> holds an HF checkpoint: config.json and *.safetensors. Tokenizing text
// is not built in yet; produce ids with the model's tokenizer. --dump-logits
// writes the prompt's last-token logits as raw little-endian f32, which
// tools/compare_hf.py checks against HF transformers.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <vkml/vkml.hpp>

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

struct Args {
    std::string model;
    std::vector<std::int32_t> tokens;
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
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string_view flag = argv[i];
        const std::string value = argv[i + 1];
        if (flag == "--model")
            args.model = value;
        else if (flag == "--tokens")
            args.tokens = parse_tokens(value);
        else if (flag == "--generate")
            args.generate = std::stoi(value);
        else if (flag == "--context")
            args.context = std::stoll(value);
        else if (flag == "--top")
            args.top = std::stoi(value);
        else if (flag == "--dump-logits")
            args.dump_logits = value;
        else if (flag == "--device")
            args.device = value;
        else
            return false;
    }
    return argc % 2 == 1 && !args.model.empty() && !args.tokens.empty();
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
                         "usage: %s --model <dir> --tokens <id,id,...> [--generate N] "
                         "[--context N] [--top N] [--dump-logits <file>] [--device <name>]\n",
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
        vkml::Llama model = vkml::Llama::load(context, args.model, args.context);
        const vkml::LlamaConfig& c = model.config();
        std::printf(
            "model    %lld layers, hidden %lld, %lld/%lld heads, vocab %lld (%.1f s to load)\n",
            static_cast<long long>(c.num_layers), static_cast<long long>(c.hidden_size),
            static_cast<long long>(c.num_heads), static_cast<long long>(c.num_kv_heads),
            static_cast<long long>(c.vocab_size), seconds_since(start));

        start = Clock::now();
        std::vector<float> logits = model.forward(args.tokens).to_vector<float>();
        const double prefill = seconds_since(start);
        std::printf("prefill  %zu tokens in %.3f s (%.1f tokens/s)\n", args.tokens.size(), prefill,
                    double(args.tokens.size()) / prefill);
        std::printf("top logits after the prompt:\n");
        print_top(logits, args.top);

        if (!args.dump_logits.empty()) {
            std::ofstream out(args.dump_logits, std::ios::binary);
            out.write(reinterpret_cast<const char*>(logits.data()),
                      static_cast<std::streamsize>(logits.size() * sizeof(float)));
        }

        std::vector<std::int32_t> generated;
        start = Clock::now();
        for (int i = 0; i < args.generate && model.position() < model.context_length(); ++i) {
            const std::int32_t next = argmax(logits);
            generated.push_back(next);
            logits = model.forward({&next, 1}).to_vector<float>();
        }
        if (!generated.empty()) {
            const double decode = seconds_since(start);
            std::printf("decode   %zu tokens in %.3f s (%.1f tokens/s)\ngenerated",
                        generated.size(), decode, double(generated.size()) / decode);
            for (std::size_t i = 0; i < generated.size(); ++i) {
                std::printf("%s%d", i ? "," : " ", generated[i]);
            }
            std::printf("\n");
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
