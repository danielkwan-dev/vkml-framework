#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <nlohmann/json.hpp>

#include "openai.hpp"

using Catch::Matchers::ContainsSubstring;
using nlohmann::json;
using namespace vkml_openai;

TEST_CASE("parse_chat_request reads messages and sampling settings", "[openai]") {
    const ChatRequest r = parse_chat_request(json::parse(R"({
        "model": "anything",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": [{"type": "text", "text": "Hi "},
                                         {"type": "text", "text": "there"}]}],
        "temperature": 0.2, "top_p": 0.9, "top_k": 40, "min_p": 0.05,
        "repetition_penalty": 1.1, "seed": 7, "max_completion_tokens": 12,
        "stream": true, "stop": "\n\n"})"));
    REQUIRE(r.messages.size() == 2);
    CHECK(r.messages[0].role == "system");
    CHECK(r.messages[1].content == "Hi there");  // text parts, joined
    CHECK(r.temperature == 0.2f);
    CHECK(r.top_p == 0.9f);
    CHECK(r.top_k == 40);
    CHECK(r.min_p == 0.05f);
    CHECK(r.repetition_penalty == 1.1f);
    CHECK(r.seed == 7u);
    CHECK(r.max_tokens == 12);
    CHECK(r.stream);
    CHECK(r.stop == std::vector<std::string>{"\n\n"});

    // Defaults, max_tokens, and a list of stop strings.
    const ChatRequest d = parse_chat_request(json::parse(
        R"({"messages": [{"role": "user", "content": "x"}], "max_tokens": 5,
            "stop": ["a", "b"]})"));
    CHECK_FALSE(d.temperature.has_value());
    CHECK_FALSE(d.stream);
    CHECK(d.max_tokens == 5);
    CHECK(d.stop == std::vector<std::string>{"a", "b"});
    CHECK(parse_chat_request(json::parse(R"({"messages": [{"role": "user", "content": "x"}]})"))
              .max_tokens == -1);
    CHECK_FALSE(d.enable_thinking.has_value());
}

TEST_CASE("parse_chat_request reads enable_thinking from chat_template_kwargs", "[openai]") {
    // As vLLM and SGLang take it, for Qwen3.
    const ChatRequest r = parse_chat_request(json::parse(
        R"({"messages": [{"role": "user", "content": "x"}],
            "chat_template_kwargs": {"enable_thinking": false}})"));
    CHECK(r.enable_thinking == false);
    const auto rejects = [](const std::string& kwargs, const std::string& what) {
        CAPTURE(kwargs);
        REQUIRE_THROWS_WITH(
            parse_chat_request(json::parse(R"({"messages": [{"role": "user", "content": "x"}],)"
                                           R"( "chat_template_kwargs": )" +
                                           kwargs + "}")),
            ContainsSubstring(what));
    };
    rejects(R"({"enable_thinking": "no"})", "enable_thinking");
    rejects(R"({"tools_style": 1})", "tools_style");
    rejects(R"([])", "chat_template_kwargs");
}

TEST_CASE("parse_chat_request rejects malformed requests with a clear message", "[openai]") {
    const auto rejects = [](const std::string& body, const std::string& what) {
        CAPTURE(body);
        REQUIRE_THROWS_WITH(parse_chat_request(json::parse(body)), ContainsSubstring(what));
    };
    rejects(R"({})", "messages");
    rejects(R"({"messages": []})", "messages");
    rejects(R"({"messages": [{"role": "user"}]})", "content");
    rejects(R"({"messages": [{"content": "x"}]})", "role");
    rejects(R"({"messages": [{"role": "user", "content": 3}]})", "content");
    rejects(R"({"messages": [{"role": "user", "content": "x"}], "temperature": "hot"})",
            "temperature");
    rejects(R"({"messages": [{"role": "user", "content": "x"}], "n": 2})", "n");
    rejects(R"({"messages": [{"role": "user", "content": "x"}], "max_tokens": 0})", "max_tokens");
}

TEST_CASE("check_stops finds a stop string and holds back text that may become one", "[openai]") {
    const std::vector<std::string> stops{"\n\nUser:", "END"};
    // No stop yet, and no ending that starts one: all of it can go out.
    CHECK(check_stops("Hello there", stops).safe == 11);
    CHECK_FALSE(check_stops("Hello there", stops).stopped);
    // "EN" may become "END", "\n\nUs" may become "\n\nUser:": hold them back.
    CHECK(check_stops("Hello EN", stops).safe == 6);
    CHECK(check_stops("Hi\n\nUs", stops).safe == 2);
    // A stop string ends the reply where it starts.
    const StopCheck done = check_stops("Hi END and more", stops);
    CHECK(done.stopped);
    CHECK(done.safe == 3);
    // The earliest of several.
    CHECK(check_stops("aEND\n\nUser:", stops).safe == 1);
    CHECK(check_stops("anything", {}).safe == 8);
}

TEST_CASE("Responses and stream chunks have OpenAI's shape", "[openai]") {
    const json full = completion_json("chatcmpl-1", 1700000000, "m", "Paris.", "stop", 10, 3);
    CHECK(full["object"] == "chat.completion");
    CHECK(full["choices"][0]["message"]["role"] == "assistant");
    CHECK(full["choices"][0]["message"]["content"] == "Paris.");
    CHECK(full["choices"][0]["finish_reason"] == "stop");
    CHECK(full["usage"]["total_tokens"] == 13);

    const json delta = chunk_json("chatcmpl-1", 1700000000, "m", json{{"content", "Pa"}}, nullptr);
    CHECK(delta["object"] == "chat.completion.chunk");
    CHECK(delta["choices"][0]["delta"]["content"] == "Pa");
    CHECK(delta["choices"][0]["finish_reason"].is_null());
    const json last = chunk_json("chatcmpl-1", 1700000000, "m", json::object(), "length");
    CHECK(last["choices"][0]["finish_reason"] == "length");

    CHECK(error_json("bad", "invalid_request_error")["error"]["message"] == "bad");
}
