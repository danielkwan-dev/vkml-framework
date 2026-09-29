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
                throw BadRequest("message \"content\" parts must be {\"type\": \"text\", \"text\": ...}");
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
        } else if (s.is_array() && std::ranges::all_of(s, [](const json& x) { return x.is_string(); })) {
            for (const json& x : s) r.stop.push_back(x.get<std::string>());
        } else {
            throw BadRequest("\"stop\" must be a string or a list of strings");
        }
        std::erase(r.stop, std::string());
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

inline json completion_json(const std::string& id, std::int64_t created, const std::string& model,
                            const std::string& content, const std::string& finish_reason,
                            std::int64_t prompt_tokens, std::int64_t completion_tokens) {
    return {{"id", id},
            {"object", "chat.completion"},
            {"created", created},
            {"model", model},
            {"choices", json::array({{{"index", 0},
                                      {"message", {{"role", "assistant"}, {"content", content}}},
                                      {"finish_reason", finish_reason}}})},
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
    return {{"error", {{"message", message}, {"type", type}, {"param", nullptr}, {"code", nullptr}}}};
}

}  // namespace vkml_openai
