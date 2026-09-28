// Chats with a LLaMA-architecture chat model: each message is formatted with
// the model's own chat template, and the reply streams as it is generated.
//
//   vkml-chat --model <dir> [--system "You are ..."] [--temperature 0.7]
//             [--top-k 0] [--top-p 0.9] [--seed N] [--max-reply 512]
//             [--context 2048] [--q8] [--device <name substring>]
//   vkml-chat --model <dir> --render-only < messages.json
//
// <dir> holds an HF checkpoint with tokenizer.json and a tokenizer_config.json
// that has a chat_template. Type /reset to start a new conversation and /quit
// (or end of input) to leave. Earlier turns stay in the KV cache, so each turn
// processes only its new tokens.
//
// --render-only reads a JSON list of {"role", "content"} messages and prints
// the formatted prompt as a JSON string, for tools/compare_chat_template.py.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>
#include <vkml/vkml.hpp>

#include "text_stream.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct Args {
    std::string model;
    std::string system;
    std::string device;
    std::int64_t context = 2048;
    int max_reply = 512;
    bool render_only = false;
    bool q8 = false;
    vkml::SamplingOptions sampling{
        .temperature = 0.7f, .top_p = 0.9f, .seed = std::random_device{}()};
};

bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--q8") {
            args.q8 = true;
            continue;
        }
        if (flag == "--render-only") {
            args.render_only = true;
            continue;
        }
        if (i + 1 == argc) return false;  // every other flag takes a value
        const std::string value = argv[++i];
        if (flag == "--model") {
            args.model = value;
        } else if (flag == "--system") {
            args.system = value;
        } else if (flag == "--device") {
            args.device = value;
        } else if (flag == "--context") {
            args.context = std::stoll(value);
        } else if (flag == "--max-reply") {
            args.max_reply = std::stoi(value);
        } else if (flag == "--temperature") {
            args.sampling.temperature = std::stof(value);
        } else if (flag == "--top-k") {
            args.sampling.top_k = std::stoi(value);
        } else if (flag == "--top-p") {
            args.sampling.top_p = std::stof(value);
        } else if (flag == "--seed") {
            args.sampling.seed = std::stoull(value);
        } else {
            return false;
        }
    }
    return !args.model.empty();
}

int render_only(const vkml::ChatTemplate& chat) {
    // Read all of stdin first: parsing straight from std::cin through stream
    // iterators never finishes with MinGW's stdio-synced streams.
    std::ostringstream text;
    text << std::cin.rdbuf();
    const auto input = nlohmann::json::parse(text.str());
    std::vector<vkml::ChatMessage> messages;
    for (const auto& m : input) messages.push_back({m.at("role"), m.at("content")});
    // As JSON, so the exact text survives the console's newline handling.
    std::cout << nlohmann::json(chat.render(messages, true)).dump() << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    try {
        if (!parse_args(argc, argv, args)) {
            std::fprintf(
                stderr,
                "usage: %s --model <dir> [--system <text>] [--temperature T] [--top-k K] "
                "[--top-p P] [--seed N] [--max-reply N] [--context N] [--q8] [--device <name>]\n"
                "       %s --model <dir> --render-only < messages.json\n",
                argv[0], argv[0]);
            return 2;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bad argument: %s\n", e.what());
        return 2;
    }
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);  // replies are UTF-8
    SetConsoleCP(CP_UTF8);
#endif

    try {
        const std::filesystem::path dir = args.model;
        const vkml::ChatTemplate chat = vkml::ChatTemplate::load(dir);
        if (args.render_only) return render_only(chat);
        const vkml::Tokenizer tokenizer{dir / "tokenizer.json"};

        vkml::ContextOptions options;
        options.device_name = args.device;
        vkml::Context context{options};
        const auto config = vkml::LlamaConfig::from_json(dir / "config.json");
        vkml::Llama model =
            vkml::Llama::load(context, dir, std::min(args.context, config.max_positions),
                              vkml::LlamaOptions{.quantize_q8 = args.q8});
        vkml::Sampler sampler{args.sampling};

        // Generation ends at the config's end-of-sequence tokens or the template's.
        std::vector<std::int32_t> stop = model.config().eos_token_ids;
        if (const auto id = tokenizer.token_id(chat.eos_token())) stop.push_back(*id);

        std::printf("%s on %s. /reset starts over, /quit leaves.\n",
                    dir.filename().string().c_str(), context.device_info().name.c_str());

        std::vector<vkml::ChatMessage> history;
        if (!args.system.empty()) history.push_back({"system", args.system});
        std::vector<std::int32_t> cached;  // the tokens the model has processed, in order

        std::string line;
        for (;;) {
            std::printf("\n> ");
            std::fflush(stdout);
            if (!std::getline(std::cin, line) || line == "/quit") break;
            if (line == "/reset") {
                history.resize(args.system.empty() ? 0 : 1);
                cached.clear();
                model.reset();
                std::printf("(new conversation)\n");
                continue;
            }
            if (line.empty()) continue;
            history.push_back({"user", line});

            // HF's apply_chat_template adds no special tokens of its own: the
            // template writes the ones the model expects.
            std::vector<std::int32_t> ids = tokenizer.encode(chat.render(history, true), false);
            if (std::int64_t(ids.size()) + args.max_reply > model.context_length()) {
                std::printf("(the conversation no longer fits the context; starting over)\n");
                history.erase(history.begin() + (args.system.empty() ? 0 : 1), history.end() - 1);
                ids = tokenizer.encode(chat.render(history, true), false);
                cached.clear();
                if (std::int64_t(ids.size()) >= model.context_length()) {
                    std::printf("(that message alone is too long for the context)\n");
                    history.pop_back();
                    continue;
                }
            }

            // Only tokens after the part the cache already holds are processed.
            std::size_t common = std::size_t(std::ranges::mismatch(cached, ids).in2 - ids.begin());
            common = std::min(common, ids.size() - 1);  // forward at least one token for logits
            model.rewind(std::int64_t(common));
            const auto start = Clock::now();
            std::vector<float> logits =
                model.forward(std::span(ids).subspan(common)).to_vector<float>();
            cached = ids;

            std::vector<std::int32_t> reply;
            std::size_t printed = 0;
            while (std::int64_t(reply.size()) < args.max_reply &&
                   model.position() < model.context_length()) {
                const std::int32_t next = sampler.sample(logits);
                if (std::ranges::find(stop, next) != stop.end()) break;
                reply.push_back(next);
                const std::string text = tokenizer.decode(reply);
                const std::size_t ready = complete_utf8_prefix(text);
                if (ready > printed) {
                    std::printf("%s", text.substr(printed, ready - printed).c_str());
                    std::fflush(stdout);
                    printed = ready;
                }
                logits = model.forward({&next, 1}).to_vector<float>();
                cached.push_back(next);
            }
            const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
            std::printf("\n  [%zu tokens, %.1f tokens/s]\n", reply.size(),
                        double(reply.size()) / seconds);
            history.push_back({"assistant", tokenizer.decode(reply)});
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
