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
// presence_penalty, seed, max_tokens (or
// max_completion_tokens), stream, stop and chat_template_kwargs'
// enable_thinking (false asks Qwen3 to answer without reasoning first, as
// --no-think does for requests that do not say); sampling they leave out defaults
// to the model's generation_config.json, then to temperature 0.7 and top-p
// 0.9, as vkml-chat. The model field is ignored: one model is served.
//
// With --reasoning-content, a reply's leading <think> ... </think> block (or
// the rest of one the chat template opened) goes in the message's
// reasoning_content rather than its content, as DeepSeek's API and vLLM's
// reasoning parsers return it; streamed, in deltas of reasoning_content.
//
// With --api-key, requests must carry "Authorization: Bearer <key>" (as the
// openai clients send their api_key); /health stays open.
//
// One model generates one reply at a time: requests wait their turn. The KV
// cache is kept between requests, so a conversation's next turn processes only
// its new tokens.

// httplib first: it includes winsock2.h, which must come before windows.h.
#include <httplib.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <exception>
#include <filesystem>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>
#include <vkml/vkml.hpp>

#include "generation.hpp"
#include "openai.hpp"

namespace {

using vkml_openai::json;
using namespace vkml_openai;
using Clock = std::chrono::steady_clock;

struct Args {
    std::string model;
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string device;
    std::int64_t context = 4096;
    std::optional<vkml::QuantType> quantize;
    bool kv_f16 = false;
    std::string api_key;
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
struct Reply {
    std::string reasoning;  // with --reasoning-content
    std::string content;
    std::string finish_reason;  // "stop", "length" or "tool_calls"
    std::size_t tokens = 0;
    json tool_calls = json::array();
};

class Server {
public:
    Server(vkml_apps::ChatModel& model, std::string name, std::optional<bool> enable_thinking,
           bool reasoning)
        : model_(model),
          name_(std::move(name)),
          enable_thinking_(enable_thinking),
          reasoning_(reasoning) {}

    // POST /v1/chat/completions, or with completion /v1/completions: the
    // prompt as given (with the tokenizer's BOS), not through the template.
    void handle_chat(const httplib::Request& req, httplib::Response& res, bool completion) {
        ChatRequest request;
        try {
            request = parse_chat_request(json::parse(req.body), completion);
        } catch (const json::exception& e) {
            return error(res, 400, std::string("the body is not valid JSON: ") + e.what());
        } catch (const BadRequest& e) {
            return error(res, 400, e.what());
        }
        std::vector<vkml::ChatMessage> messages;
        for (const Message& m : request.messages) {
            messages.push_back({m.role, m.content, m.fields.is_null() ? "" : m.fields.dump()});
        }
        std::vector<std::int32_t> prompt;
        bool thinking = false;  // the template opened a reasoning block
        try {
            if (request.prompt) prompt = model_.tokenizer.encode(*request.prompt, true);
        } catch (const std::exception& e) {
            return error(res, 400, std::string("the prompt cannot be encoded: ") + e.what());
        }
        if (!request.prompt) try {
                // As ChatModel::encode, keeping the text to look at its end.
                const std::string text = model_.chat.render(
                    messages, true,
                    request.enable_thinking ? request.enable_thinking : enable_thinking_,
                    request.tools.is_null() ? "" : request.tools.dump());
                prompt = model_.tokenizer.encode(text, false);
                thinking = reasoning_ && opens_reasoning(text);
            } catch (const std::exception& e) {
                return error(res, 400, std::string("the chat template failed: ") + e.what());
            }
        if (std::int64_t(prompt.size()) >= model_.model.context_length()) {
            return error(res, 400,
                         "the messages are " + std::to_string(prompt.size()) +
                             " tokens, and the context holds " +
                             std::to_string(model_.model.context_length()));
        }

        const std::string id = (completion ? "cmpl-" : "chatcmpl-") + std::to_string(next_id_++);
        // 8 letters and digits, which the call's index makes 9 (as Mistral's
        // template needs): "call" and the request's number in base 36.
        request.call_id = "call";
        for (std::uint64_t n = next_id_ - 1, i = 0; i < 4; ++i, n /= 36) {
            request.call_id.insert(4, 1, "0123456789abcdefghijklmnopqrstuvwxyz"[n % 36]);
        }
        const std::int64_t created = std::int64_t(std::time(nullptr));
        if (!request.stream) {
            const Reply reply = generate(request, prompt, thinking,
                                         [](const char*, std::string_view) { return true; });
            const auto p = std::int64_t(prompt.size()), r = std::int64_t(reply.tokens);
            if (completion) {
                const json usage = {
                    {"prompt_tokens", p}, {"completion_tokens", r}, {"total_tokens", p + r}};
                return res.set_content(
                    text_json(id, created, name_, reply.content, reply.finish_reason, usage).dump(),
                    "application/json");
            }
            res.set_content(
                completion_json(id, created, name_, reply.content, reply.finish_reason,
                                std::int64_t(prompt.size()), std::int64_t(reply.tokens),
                                reasoning_ ? std::optional(reply.reasoning) : std::nullopt,
                                reply.tool_calls)
                    .dump(),
                "application/json");
            return;
        }
        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider(
            "text/event-stream", [this, request, prompt, thinking, id, created, completion](
                                     std::size_t, httplib::DataSink& sink) {
                const auto event = [&](const json& data) {
                    const std::string line = "data: " + data.dump() + "\n\n";
                    return sink.write(line.data(), line.size());
                };
                if (!completion &&
                    !event(chunk_json(id, created, name_, {{"role", "assistant"}, {"content", ""}},
                                      nullptr))) {
                    return false;
                }
                try {
                    const Reply reply = generate(
                        request, prompt, thinking, [&](const char* field, std::string_view piece) {
                            return event(
                                completion
                                    ? text_json(id, created, name_, std::string(piece), nullptr)
                                    : chunk_json(id, created, name_, {{field, piece}}, nullptr));
                        });
                    if (!reply.tool_calls.empty()) {
                        json calls = reply.tool_calls;
                        for (std::size_t i = 0; i < calls.size(); ++i) calls[i]["index"] = i;
                        event(chunk_json(id, created, name_, {{"tool_calls", calls}}, nullptr));
                    }
                    event(completion ? text_json(id, created, name_, "", reply.finish_reason)
                                     : chunk_json(id, created, name_, json::object(),
                                                  reply.finish_reason));
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "error: %s\n", e.what());
                    event(error_json(e.what(), "server_error"));
                }
                const std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();
                return true;
            });
    }

    void handle_models(const httplib::Request&, httplib::Response& res) const {
        const json list = {
            {"object", "list"},
            {"data",
             json::array(
                 {{{"id", name_}, {"object", "model"}, {"created", 0}, {"owned_by", "vkml"}}})}};
        res.set_content(list.dump(), "application/json");
    }

    static void error(httplib::Response& res, int status, const std::string& message) {
        res.status = status;
        res.set_content(
            error_json(message, status < 500 ? "invalid_request_error" : "server_error").dump(),
            "application/json");
    }

private:
    // Generates the reply to prompt, handing emit its text piece by piece as
    // it becomes final, with the field it belongs in ("content", or with
    // --reasoning-content "reasoning_content"); emit returns false when the
    // client has gone. thinking: the prompt left the reply inside a reasoning
    // block.
    template <class Emit>
    Reply generate(const ChatRequest& request, const std::vector<std::int32_t>& prompt,
                   bool thinking, Emit&& emit) {
        vkml::SamplingOptions o{.temperature = 0.7f, .top_p = 0.9f};
        if (model_.generation) o = model_.generation->apply(o);
        if (request.temperature) o.temperature = *request.temperature;
        if (request.top_k) o.top_k = *request.top_k;
        if (request.top_p) o.top_p = *request.top_p;
        if (request.min_p) o.min_p = *request.min_p;
        if (request.repetition_penalty) o.repetition_penalty = *request.repetition_penalty;
        if (request.frequency_penalty) o.frequency_penalty = *request.frequency_penalty;
        if (request.presence_penalty) o.presence_penalty = *request.presence_penalty;
        o.seed = request.seed ? *request.seed : std::random_device{}();
        vkml::Sampler sampler{o};

        const std::lock_guard lock(mutex_);
        const auto start = Clock::now();
        Reply reply;
        std::size_t sent_reasoning = 0, sent_content = 0;
        bool stopped = false;  // at a stop string
        // Sends text up to where it is final, and says whether to go on.
        const auto send = [&](const std::string& text, bool last) {
            const StopCheck check = check_stops(text, request.stop);
            stopped = check.stopped;
            const std::size_t end = check.stopped || !last ? check.safe : text.size();
            const std::string_view out = std::string_view(text).substr(0, end);
            ReasoningSplit split = reasoning_ ? split_reasoning(out, thinking, last || stopped)
                                              : ReasoningSplit{"", std::string(out)};
            // With tools, text from a tool call on is held back: the calls
            // go out whole at the end.
            std::string_view shown = split.content;
            std::string after;  // text after calls opening the reply
            if (!request.tools.is_null()) {
                shown = shown.substr(0, shown.find(vkml_openai::kToolCallOpen));
                if (!last)
                    shown.remove_suffix(detail::partial_tag(shown, vkml_openai::kToolCallOpen));
                // A reply opening with "{" or "[" may be a Llama 3.x or
                // Mistral call: held to its end, then let out if it is not
                // one, or only what follows the calls.
                const std::string_view opening = detail::trim_start(shown);
                if (opening.starts_with('{') || opening.starts_with('[')) {
                    if (!last) {
                        shown = {};
                    } else if (ToolCalls c = parse_tool_calls(shown, ""); !c.calls.empty()) {
                        after = std::move(c.content);
                        shown = after;
                    }
                }
            }
            bool open = true;
            if (split.reasoning.size() > sent_reasoning) {
                open = emit("reasoning_content",
                            std::string_view(split.reasoning).substr(sent_reasoning));
                sent_reasoning = split.reasoning.size();
            }
            if (open && shown.size() > sent_content) {
                open = emit("content", shown.substr(sent_content));
                sent_content = shown.size();
            }
            reply.reasoning = std::move(split.reasoning);
            reply.content = std::move(split.content);
            return open && !check.stopped;
        };
        std::vector<std::int32_t> tokens;
        bool open = true;
        const auto finish = model_.generate(prompt, sampler, request.max_tokens, tokens,
                                            [&](const std::string& text) {
                                                open = send(text, false);
                                                return open;
                                            });
        // Text held back in case it began a stop string is final now.
        if (open) send(model_.tokenizer.decode(tokens), true);
        reply.tokens = tokens.size();
        reply.finish_reason =
            stopped || finish == vkml_apps::ChatModel::Finish::stop ? "stop" : "length";
        if (!request.tools.is_null()) {
            ToolCalls calls = parse_tool_calls(reply.content, request.call_id);
            if (!calls.calls.empty()) {
                reply.content = std::move(calls.content);
                reply.tool_calls = std::move(calls.calls);
                reply.finish_reason = "tool_calls";
            }
        }
        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        std::printf("%zu prompt tokens, %zu reply tokens in %.2f s (%.1f tokens/s), %s\n",
                    prompt.size(), tokens.size(), seconds, double(tokens.size()) / seconds,
                    reply.finish_reason.c_str());
        std::fflush(stdout);
        return reply;
    }

    vkml_apps::ChatModel& model_;
    std::string name_;
    std::optional<bool> enable_thinking_;  // for requests that do not say
    bool reasoning_;                       // --reasoning-content
    std::mutex mutex_;
    std::atomic<std::uint64_t> next_id_{1};
};

}  // namespace

int main(int argc, char** argv) {
    Args args;
    try {
        if (!parse_args(argc, argv, args)) {
            std::fprintf(
                stderr,
                "usage: %s --model <dir or .gguf> [--host 127.0.0.1] [--port 8080] "
                "[--context N] [--q8 | --q4] [--kv-f16] [--device <name>] "
                "[--api-key <key>] [--no-think] [--reasoning-content] [--chat-template <dir>]\n",
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
        // Browser chat UIs call from their own origin.
        http.set_default_headers({{"Access-Control-Allow-Origin", "*"},
                                  {"Access-Control-Allow-Headers", "*"},
                                  {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"}});
        if (!args.api_key.empty()) {
            http.set_pre_routing_handler([&](const httplib::Request& req, httplib::Response& res) {
                if (req.method == "OPTIONS" || req.path == "/health" ||
                    req.get_header_value("Authorization") == "Bearer " + args.api_key) {
                    return httplib::Server::HandlerResponse::Unhandled;
                }
                Server::error(res, 401, "a valid API key is needed");
                return httplib::Server::HandlerResponse::Handled;
            });
        }
        http.Options(".*", [](const httplib::Request&, httplib::Response&) {});
        http.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            res.set_content(R"({"status":"ok"})", "application/json");
        });
        http.Get("/v1/models", [&](const httplib::Request& req, httplib::Response& res) {
            server.handle_models(req, res);
        });
        http.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
            server.handle_chat(req, res, false);
        });
        http.Post("/v1/completions", [&](const httplib::Request& req, httplib::Response& res) {
            server.handle_chat(req, res, true);
        });
        http.set_exception_handler(
            [](const httplib::Request&, httplib::Response& res, const std::exception_ptr& ep) {
                std::string what = "unknown error";
                try {
                    std::rethrow_exception(ep);
                } catch (const std::exception& e) {
                    what = e.what();
                } catch (...) {
                }
                std::fprintf(stderr, "error: %s\n", what.c_str());
                Server::error(res, 500, what);
            });

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
