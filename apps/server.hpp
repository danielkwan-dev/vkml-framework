#pragma once

// vkml-server's request handling and routes, apart from its main so tests can
// run them: an OpenAI-compatible server over one ChatModel (see
// vkml_server.cpp).

// httplib first, and this header first in a file that includes it: httplib
// includes winsock2.h, which must come before windows.h, and on MinGW sets the
// Windows version before any system header does.
#include <httplib.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <exception>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>
#include <vkml/vkml.hpp>

#include "generation.hpp"
#include "openai.hpp"

namespace vkml_server {

using vkml_openai::json;
using namespace vkml_openai;
using Clock = std::chrono::steady_clock;

struct Reply {
    std::string reasoning;  // with --reasoning-content
    std::string content;
    std::string finish_reason;  // "stop", "length" or "tool_calls"
    std::size_t tokens = 0;
    json tool_calls = json::array();
    std::vector<LogprobEntry> logprobs;  // if asked for: up to a stop string
    std::size_t logprobs_streamed = 0;   // how many chunks have carried
    // With echo: the prompt's text, and its tokens' logprobs if asked for.
    std::string echoed;
    std::vector<LogprobEntry> prompt_logprobs;
};

// A request's logprobs in its API's shape, or null if it asked for none or
// there are none.
inline json logprobs_json(const ChatRequest& request, std::span<const LogprobEntry> entries) {
    if (request.logprobs < 0 || entries.empty()) return nullptr;
    const std::vector<LogprobEntry> list(entries.begin(), entries.end());
    return request.completion ? text_logprobs(list) : chat_logprobs(list);
}

// All of a reply's logprobs: an echoed prompt's, then the generated tokens'.
inline std::vector<LogprobEntry> all_logprobs(const Reply& reply) {
    std::vector<LogprobEntry> all = reply.prompt_logprobs;
    all.insert(all.end(), reply.logprobs.begin(), reply.logprobs.end());
    return all;
}

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
            sampling_options(request, defaults());  // only to check it
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
        if (!request.prompt_ids.empty()) {
            const auto vocab = std::int32_t(model_.model.config().vocab_size);
            if (std::ranges::any_of(request.prompt_ids,
                                    [&](std::int32_t i) { return i >= vocab; })) {
                return error(
                    res, 400,
                    "the prompt has token ids past the vocabulary of " + std::to_string(vocab));
            }
            prompt = request.prompt_ids;
        }
        if (!request.completion) try {
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
            const Reply reply =
                generate(request, prompt, thinking,
                         [](const char*, std::string_view, const json&) { return true; });
            const auto p = std::int64_t(prompt.size()), r = std::int64_t(reply.tokens);
            json out;
            if (completion) {
                const json usage = {
                    {"prompt_tokens", p}, {"completion_tokens", r}, {"total_tokens", p + r}};
                out = text_json(id, created, name_, reply.echoed + reply.content,
                                reply.finish_reason, usage);
            } else {
                out = completion_json(id, created, name_, reply.content, reply.finish_reason, p, r,
                                      reasoning_ ? std::optional(reply.reasoning) : std::nullopt,
                                      reply.tool_calls);
            }
            if (request.logprobs >= 0) {
                out["choices"][0]["logprobs"] =
                    completion ? text_logprobs(all_logprobs(reply)) : chat_logprobs(reply.logprobs);
            }
            res.set_content(out.dump(), "application/json");
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
                    // A chunk, with the log probabilities of its tokens if asked.
                    const auto with = [](json chunk, const json& logprobs) {
                        if (!logprobs.is_null()) chunk["choices"][0]["logprobs"] = logprobs;
                        return chunk;
                    };
                    const Reply reply = generate(
                        request, prompt, thinking,
                        [&](const char* field, std::string_view piece, const json& lp) {
                            return event(with(
                                completion
                                    ? text_json(id, created, name_, std::string(piece), nullptr)
                                    : chunk_json(id, created, name_, {{field, piece}}, nullptr),
                                lp));
                        });
                    if (!reply.tool_calls.empty()) {
                        json calls = reply.tool_calls;
                        for (std::size_t i = 0; i < calls.size(); ++i) calls[i]["index"] = i;
                        event(chunk_json(id, created, name_, {{"tool_calls", calls}}, nullptr));
                    }
                    // Tokens no chunk has carried (held back, or with no text) go
                    // with the last.
                    const json rest = logprobs_json(
                        request, std::span(reply.logprobs).subspan(reply.logprobs_streamed));
                    event(with(completion ? text_json(id, created, name_, "", reply.finish_reason)
                                          : chunk_json(id, created, name_, json::object(),
                                                       reply.finish_reason),
                               rest));
                    if (request.include_usage) {
                        event(usage_chunk_json(id, created, name_, completion,
                                               std::int64_t(prompt.size()),
                                               std::int64_t(reply.tokens)));
                    }
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
    // Sampling for settings a request leaves out: the model's suggestions,
    // else temperature 0.7 and top-p 0.9.
    vkml::SamplingOptions defaults() const {
        const vkml::SamplingOptions o{.temperature = 0.7f, .top_p = 0.9f};
        return model_.generation ? model_.generation->apply(o) : o;
    }

    // Generates the reply to prompt, handing emit its text piece by piece as
    // it becomes final, with the field it belongs in ("content", or with
    // --reasoning-content "reasoning_content") and the log probabilities not
    // yet sent (or null); emit returns false when the client has gone. thinking: the prompt left
    // the reply inside a reasoning block.
    template <class Emit>
    Reply generate(const ChatRequest& request, const std::vector<std::int32_t>& prompt,
                   bool thinking, Emit&& emit) {
        vkml::SamplingOptions o = defaults();
        o.seed = std::random_device{}();  // unless the request gives one
        vkml::Sampler sampler{sampling_options(request, o)};

        const std::lock_guard lock(mutex_);
        const auto start = Clock::now();
        Reply reply;
        // echo: the prompt's text, and its tokens' logprobs, go first.
        if (request.echo) {
            reply.echoed = model_.tokenizer.decode(prompt);
            if (request.logprobs >= 0) {
                for (const auto& t : model_.score(prompt, request.logprobs)) {
                    LogprobEntry e{t.text, t.logprobs.logprob, t.offset, {}};
                    for (const auto& [token, logprob] : t.logprobs.top) {
                        e.top.emplace_back(model_.tokenizer.token_text(token), logprob);
                    }
                    reply.prompt_logprobs.push_back(std::move(e));
                }
            }
            if (!reply.echoed.empty() || !reply.prompt_logprobs.empty()) {
                emit("content", reply.echoed, logprobs_json(request, reply.prompt_logprobs));
            }
        }
        const std::size_t shift = reply.echoed.size();  // generated text comes after it
        std::size_t sent_reasoning = 0, sent_content = 0;
        bool stopped = false;  // at a stop string
        std::vector<vkml_apps::ChatModel::TokenLogprob> token_logprobs;
        // The log probabilities no chunk has carried yet, for the next one.
        const auto unsent = [&] {
            const json lp =
                logprobs_json(request, std::span(reply.logprobs).subspan(reply.logprobs_streamed));
            reply.logprobs_streamed = reply.logprobs.size();
            return lp;
        };
        // Sends text up to where it is final, and says whether to go on.
        const auto send = [&](const std::string& text, bool last) {
            const StopCheck check = check_stops(text, request.stop);
            stopped = check.stopped;
            const std::size_t end = check.stopped || !last ? check.safe : text.size();
            const std::string_view out = std::string_view(text).substr(0, end);
            // Entries for the tokens whose text starts before end: at the
            // end, all but those of a stop string.
            const std::size_t limit = last && !check.stopped ? text.size() + 1 : end;
            while (reply.logprobs.size() < token_logprobs.size() &&
                   token_logprobs[reply.logprobs.size()].offset < limit) {
                const auto& t = token_logprobs[reply.logprobs.size()];
                LogprobEntry e{t.text, t.logprobs.logprob, shift + t.offset, {}};
                for (const auto& [token, logprob] : t.logprobs.top) {
                    e.top.emplace_back(model_.tokenizer.token_text(token), logprob);
                }
                reply.logprobs.push_back(std::move(e));
            }
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
                            std::string_view(split.reasoning).substr(sent_reasoning), unsent());
                sent_reasoning = split.reasoning.size();
            }
            if (open && shown.size() > sent_content) {
                open = emit("content", shown.substr(sent_content), unsent());
                sent_content = shown.size();
            }
            reply.reasoning = std::move(split.reasoning);
            reply.content = std::move(split.content);
            return open && !check.stopped;
        };
        std::vector<std::int32_t> tokens;
        bool open = true;
        const auto finish = model_.generate(
            prompt, sampler, request.max_tokens, tokens,
            [&](const std::string& text) {
                open = send(text, false);
                return open;
            },
            request.logprobs >= 0 ? &token_logprobs : nullptr, std::max(request.logprobs, 0));
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

// Serves server's routes on http; with api_key, requests but /health need it.
inline void add_routes(httplib::Server& http, Server& server, const std::string& api_key) {
    // Browser chat UIs call from their own origin.
    http.set_default_headers({{"Access-Control-Allow-Origin", "*"},
                              {"Access-Control-Allow-Headers", "*"},
                              {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"}});
    if (!api_key.empty()) {
        http.set_pre_routing_handler(
            [api_key](const httplib::Request& req, httplib::Response& res) {
                if (req.method == "OPTIONS" || req.path == "/health" ||
                    req.get_header_value("Authorization") == "Bearer " + api_key) {
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
    http.Get("/v1/models", [&server](const httplib::Request& req, httplib::Response& res) {
        server.handle_models(req, res);
    });
    http.Post("/v1/chat/completions",
              [&server](const httplib::Request& req, httplib::Response& res) {
                  server.handle_chat(req, res, false);
              });
    http.Post("/v1/completions", [&server](const httplib::Request& req, httplib::Response& res) {
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
}

}  // namespace vkml_server
