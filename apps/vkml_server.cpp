// Serves a chat model over OpenAI's chat completions API, so existing clients,
// chat UIs and scripts can use vkml:
//
//   vkml-server --model <dir or .gguf> [--host 127.0.0.1] [--port 8080]
//               [--context 4096] [--q8 | --q4] [--kv-f16] [--device <name substring>]
//               [--api-key <key>] [--no-think] [--reasoning-content]
//
//   curl http://127.0.0.1:8080/v1/chat/completions
//        -d '{"messages": [{"role": "user", "content": "Hi!"}], "stream": true}'
//
// Routes: POST /v1/chat/completions (streamed as server-sent events with
// "stream": true), GET /v1/models and GET /health. Requests take messages,
// temperature, top_p, top_k, min_p, repetition_penalty, frequency_penalty,
// presence_penalty, seed, max_tokens (or max_completion_tokens), stream,
// stream_options' include_usage, stop and chat_template_kwargs'
// enable_thinking (false asks Qwen3 to answer without reasoning first, as
// --no-think does for requests that do not say), logprobs (chat's
// top_logprobs) and a completion's echo; response_format other than text is
// refused. Sampling they leave out
// defaults to the model's generation_config.json, then to temperature 0.7 and top-p 0.9, as
// vkml-chat. The model field is ignored: one model is served.
//
// With --reasoning-content, a reply's leading <think> ... </think> block (or
// the rest of one the chat template opened) goes in the message's
// reasoning_content rather than its content, as DeepSeek's API and vLLM's
// reasoning parsers return it; streamed, in deltas of reasoning_content.
//
// With --api-key, requests must carry "Authorization: Bearer <key>" (as the
// openai clients send their api_key); /health stays open. The server speaks
// plain HTTP: beyond this machine, put it behind a TLS proxy.
//
// Browsers read replies only from --cors-origin (say http://localhost:3000, a
// chat UI's), none by default, so web pages cannot use the server.
//
// One model generates one reply at a time: requests wait their turn. The KV
// cache is kept between requests, so a conversation's next turn processes only
// its new tokens.

// First: it includes httplib, which must come before any system header.
#include "server.hpp"

#include <cstdio>
#include <exception>
#include <filesystem>
#include <optional>
#include <string>

namespace {

using vkml_server::Server;

struct Args {
    std::string model;
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string device;
    std::int64_t context = 4096;
    std::optional<vkml::QuantType> quantize;
    bool kv_f16 = false;
    std::string api_key;
    std::string cors_origin;
    std::optional<bool> enable_thinking;  // --no-think: false
    bool reasoning_content = false;
    std::string chat_template;  // a model directory whose template to use
};

bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--q8" || flag == "--q4") {
            args.quantize = flag == "--q8" ? vkml::QuantType::q8_0 : vkml::QuantType::q4_0;
            continue;
        }
        if (flag == "--kv-f16") {
            args.kv_f16 = true;
            continue;
        }
        if (flag == "--no-think") {
            args.enable_thinking = false;
            continue;
        }
        if (flag == "--reasoning-content") {
            args.reasoning_content = true;
            continue;
        }
        if (i + 1 == argc) return false;  // every other flag takes a value
        const std::string value = argv[++i];
        if (flag == "--model") {
            args.model = value;
        } else if (flag == "--host") {
            args.host = value;
        } else if (flag == "--port") {
            args.port = std::stoi(value);
        } else if (flag == "--chat-template") {
            args.chat_template = value;
        } else if (flag == "--api-key") {
            args.api_key = value;
        } else if (flag == "--cors-origin") {
            args.cors_origin = value;
        } else if (flag == "--device") {
            args.device = value;
        } else if (flag == "--context") {
            args.context = std::stoll(value);
        } else {
            return false;
        }
    }
    return !args.model.empty();
}

// A finished reply.
}  // namespace

int main(int argc, char** argv) {
    Args args;
    try {
        if (!parse_args(argc, argv, args)) {
            std::fprintf(stderr,
                         "usage: %s --model <dir or .gguf> [--host 127.0.0.1] [--port 8080] "
                         "[--context N] [--q8 | --q4] [--kv-f16] [--device <name>] "
                         "[--api-key <key>] [--cors-origin <origin>] [--no-think] "
                         "[--reasoning-content] [--chat-template <dir>]\n",
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
        const std::filesystem::path path = args.model;
        auto model = vkml_apps::ChatModel::load(
            context, path, args.context,
            vkml::LlamaOptions{.quantize = args.quantize,
                               .kv_cache = args.kv_f16 ? vkml::DType::F16 : vkml::DType::F32},
            args.chat_template);
        const std::string name =
            path.extension() == ".gguf" ? path.stem().string() : path.filename().string();
        Server server{model, name, args.enable_thinking, args.reasoning_content};

        httplib::Server http;
        vkml_server::add_routes(http, server, args.api_key, args.cors_origin);
        if (args.api_key.empty() && args.host != "127.0.0.1" && args.host != "localhost" &&
            args.host != "::1") {
            std::fprintf(stderr,
                         "warning: serving on %s without --api-key: anyone who can reach this "
                         "port can use the model\n",
                         args.host.c_str());
        }

        if (!http.bind_to_port(args.host, args.port)) {
            throw vkml::Error("cannot listen on " + args.host + ":" + std::to_string(args.port));
        }
        std::printf("serving %s on %s at http://%s:%d/v1 (context %lld tokens)\n", name.c_str(),
                    context.device_info().name.c_str(), args.host.c_str(), args.port,
                    static_cast<long long>(model.model.context_length()));
        std::fflush(stdout);
        http.listen_after_bind();
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
