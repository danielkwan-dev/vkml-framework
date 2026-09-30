#pragma once

// The parts of OpenAI's chat completions API that vkml-server speaks: parsing
// a request and building responses and stream chunks, apart from any HTTP so
// they can be tested alone.

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace vkml_openai {

using nlohmann::json;

// A request the server cannot serve; its message goes back to the client.
struct BadRequest : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Message {
    std::string role;
    std::string content;
};

// POST /v1/chat/completions. Settings the request leaves out stay empty, for
// the server's defaults.
struct ChatRequest {
    std::vector<Message> messages;
    std::optional<float> temperature, top_p, min_p, repetition_penalty;
    std::optional<int> top_k;
    std::optional<std::uint64_t> seed;
    int max_tokens = -1;  // -1: until the end of the reply or of the context
    bool stream = false;
    std::vector<std::string> stop;
    // chat_template_kwargs' enable_thinking: false asks Qwen3 not to reason.
    std::optional<bool> enable_thinking;
};

namespace detail {

inline std::optional<float> number(const json& body, const char* key) {
    if (!body.contains(key) || body[key].is_null()) return std::nullopt;
    if (!body[key].is_number()) throw BadRequest(std::string("\"") + key + "\" must be a number");
    return body[key].get<float>();
}

inline std::optional<std::int64_t> integer(const json& body, const char* key) {
    if (!body.contains(key) || body[key].is_null()) return std::nullopt;
    if (!body[key].is_number_integer()) {
        throw BadRequest(std::string("\"") + key + "\" must be an integer");
    }
    return body[key].get<std::int64_t>();
}

// A message's content: a string, or a list of parts of which vkml reads the
// text ones.
inline std::string content(const json& message) {
    if (!message.contains("content")) throw BadRequest("every message needs a \"content\"");
    const json& c = message["content"];
    if (c.is_string()) return c.get<std::string>();
    if (c.is_array()) {
        std::string text;
        for (const json& part : c) {
            if (!part.is_object() || part.value("type", "") != "text" || !part.contains("text") ||
                !part["text"].is_string()) {
                throw BadRequest(
                    "message \"content\" parts must be {\"type\": \"text\", \"text\": ...}");
            }
            text += part["text"].get<std::string>();
        }
        return text;
    }
    throw BadRequest("message \"content\" must be a string or a list of text parts");
}

}  // namespace detail

inline ChatRequest parse_chat_request(const json& body) {
    if (!body.is_object()) throw BadRequest("the request body must be a JSON object");
    ChatRequest r;
    if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty()) {
        throw BadRequest("\"messages\" must be a non-empty list");
    }
    for (const json& m : body["messages"]) {
        if (!m.is_object() || !m.contains("role") || !m["role"].is_string()) {
            throw BadRequest("every message needs a string \"role\"");
        }
        r.messages.push_back({m["role"].get<std::string>(), detail::content(m)});
    }
    r.temperature = detail::number(body, "temperature");
    r.top_p = detail::number(body, "top_p");
    r.min_p = detail::number(body, "min_p");
    r.repetition_penalty = detail::number(body, "repetition_penalty");
    if (const auto k = detail::integer(body, "top_k")) r.top_k = int(*k);
    if (const auto s = detail::integer(body, "seed")) r.seed = std::uint64_t(*s);
    if (const auto n = detail::integer(body, "n"); n && *n != 1) {
        throw BadRequest("\"n\" must be 1: vkml generates one reply per request");
    }
    // max_completion_tokens is the newer name.
    auto max = detail::integer(body, "max_completion_tokens");
    if (!max) max = detail::integer(body, "max_tokens");
    if (max) {
        if (*max < 1) throw BadRequest("\"max_tokens\" must be at least 1");
        r.max_tokens = int(std::min<std::int64_t>(*max, 1 << 30));
    }
    if (body.contains("stream") && !body["stream"].is_null()) {
        if (!body["stream"].is_boolean()) throw BadRequest("\"stream\" must be true or false");
        r.stream = body["stream"].get<bool>();
    }
    if (body.contains("stop") && !body["stop"].is_null()) {
        const json& s = body["stop"];
        if (s.is_string()) {
            r.stop.push_back(s.get<std::string>());
        } else if (s.is_array() &&
                   std::ranges::all_of(s, [](const json& x) { return x.is_string(); })) {
            for (const json& x : s) r.stop.push_back(x.get<std::string>());
        } else {
            throw BadRequest("\"stop\" must be a string or a list of strings");
        }
        std::erase(r.stop, std::string());
    }
    // Variables for the chat template, as vLLM and SGLang take them; vkml
    // passes on only the one templates commonly read.
    if (body.contains("chat_template_kwargs") && !body["chat_template_kwargs"].is_null()) {
        const json& kwargs = body["chat_template_kwargs"];
        if (!kwargs.is_object()) throw BadRequest("\"chat_template_kwargs\" must be an object");
        for (const auto& [key, value] : kwargs.items()) {
            if (key != "enable_thinking") {
                throw BadRequest("chat_template_kwargs \"" + key +
                                 "\" is not supported: only \"enable_thinking\" is");
            }
            if (!value.is_null() && !value.is_boolean()) {
                throw BadRequest("chat_template_kwargs \"enable_thinking\" must be true or false");
            }
            if (value.is_boolean()) r.enable_thinking = value.get<bool>();
        }
    }
    return r;
}

// Where a reply's text may be sent up to, given stop strings: to the start of
// the first stop string in it (and then it is finished), or else short of any
// ending that the next tokens could complete into one.
struct StopCheck {
    std::size_t safe;
    bool stopped;
};

inline StopCheck check_stops(std::string_view text, const std::vector<std::string>& stops) {
    std::size_t first = std::string_view::npos;
    for (const std::string& s : stops) first = std::min(first, text.find(s));
    if (first != std::string_view::npos) return {first, true};
    std::size_t held = 0;
    for (const std::string& s : stops) {
        for (std::size_t n = std::min(s.size() - 1, text.size()); n > held; --n) {
            if (text.ends_with(std::string_view(s).substr(0, n))) {
                held = n;
                break;
            }
        }
    }
    return {text.size() - held, false};
}

// A reply split into its reasoning, the <think> ... </think> block reasoning
// models (Qwen3, DeepSeek R1) begin with, and its answer: what DeepSeek's API
// and vLLM's reasoning parsers return as reasoning_content and content.
struct ReasoningSplit {
    std::string reasoning;
    std::string content;
};

namespace detail {

inline constexpr std::string_view kThinkOpen = "<think>";
inline constexpr std::string_view kThinkClose = "</think>";

inline bool is_space(char c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }

inline std::string_view trim_start(std::string_view s) {
    while (!s.empty() && is_space(s.front())) s.remove_prefix(1);
    return s;
}

inline std::string_view trim(std::string_view s) {
    s = trim_start(s);
    while (!s.empty() && is_space(s.back())) s.remove_suffix(1);
    return s;
}

// The length of the longest ending of text that is a proper prefix of tag.
inline std::size_t partial_tag(std::string_view text, std::string_view tag) {
    for (std::size_t n = std::min(tag.size() - 1, text.size()); n > 0; --n) {
        if (text.ends_with(tag.substr(0, n))) return n;
    }
    return 0;
}

}  // namespace detail

// Splits text, a reply so far, at its reasoning block: one that opens it
// (after any whitespace), or with inside, one the prompt opened. Reasoning
// never closed is all reasoning; text that does not open one is all answer.
// The whitespace around the reasoning and before the answer is dropped.
// Unless last, text that may yet become a tag (or whitespace that may yet be
// dropped) is held back, so each split of a growing text extends the last.
inline ReasoningSplit split_reasoning(std::string_view text, bool inside, bool last) {
    using detail::kThinkClose;
    using detail::kThinkOpen;
    std::string_view rest = text;
    if (!inside) {
        const std::string_view start = detail::trim_start(text);
        if (!start.starts_with(kThinkOpen)) {
            if (!last && kThinkOpen.starts_with(start)) return {};  // it may yet
            return {"", std::string(text)};
        }
        rest = start.substr(kThinkOpen.size());
    }
    const std::size_t close = rest.find(kThinkClose);
    if (close == std::string_view::npos) {
        if (!last) rest.remove_suffix(detail::partial_tag(rest, kThinkClose));
        return {std::string(detail::trim(rest)), ""};
    }
    return {std::string(detail::trim(rest.substr(0, close))),
            std::string(detail::trim_start(rest.substr(close + kThinkClose.size())))};
}

// Whether a rendered prompt ends inside a reasoning block, as templates that
// open one for the model do (Qwen3's 2507 thinking models).
inline bool opens_reasoning(std::string_view prompt) {
    return detail::trim(prompt).ends_with(detail::kThinkOpen);
}

// With reasoning, the message carries it as reasoning_content.
inline json completion_json(const std::string& id, std::int64_t created, const std::string& model,
                            const std::string& content, const std::string& finish_reason,
                            std::int64_t prompt_tokens, std::int64_t completion_tokens,
                            const std::optional<std::string>& reasoning = std::nullopt) {
    json message = {{"role", "assistant"}, {"content", content}};
    if (reasoning) message["reasoning_content"] = *reasoning;
    return {{"id", id},
            {"object", "chat.completion"},
            {"created", created},
            {"model", model},
            {"choices",
             json::array({{{"index", 0}, {"message", message}, {"finish_reason", finish_reason}}})},
            {"usage",
             {{"prompt_tokens", prompt_tokens},
              {"completion_tokens", completion_tokens},
              {"total_tokens", prompt_tokens + completion_tokens}}}};
}

// One server-sent event of a streamed reply: delta is {"role": ...} first,
// then {"content": ...} pieces, then {} with the finish_reason.
inline json chunk_json(const std::string& id, std::int64_t created, const std::string& model,
                       const json& delta, const json& finish_reason) {
    return {{"id", id},
            {"object", "chat.completion.chunk"},
            {"created", created},
            {"model", model},
            {"choices",
             json::array({{{"index", 0}, {"delta", delta}, {"finish_reason", finish_reason}}})}};
}

inline json error_json(const std::string& message, const std::string& type) {
    return {
        {"error", {{"message", message}, {"type", type}, {"param", nullptr}, {"code", nullptr}}}};
}

}  // namespace vkml_openai
