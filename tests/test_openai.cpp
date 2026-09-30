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

TEST_CASE("split_reasoning separates a leading think block from the answer", "[openai]") {
    const auto split = [](std::string_view text, bool inside, bool last) {
        const ReasoningSplit s = split_reasoning(text, inside, last);
        return std::vector<std::string>{s.reasoning, s.content};
    };
    using V = std::vector<std::string>;
    // The whitespace around the reasoning, and between it and the answer, goes.
    CHECK(split("<think>\nLet me see.\n</think>\n\nParis.", false, true) ==
          V{"Let me see.", "Paris."});
    CHECK(split("Paris.", false, true) == V{"", "Paris."});
    CHECK(split("  Paris <think>", false, true) == V{"", "  Paris <think>"});
    CHECK(split("\n<think></think>\n\nParis.", false, true) == V{"", "Paris."});
    // Cut short while thinking, all of it is reasoning.
    CHECK(split("<think>still thinking", false, true) == V{"still thinking", ""});
    // A prompt that opened the block itself: the reply starts inside it.
    CHECK(split("Let me see.</think>Paris.", true, true) == V{"Let me see.", "Paris."});
    CHECK(split("no tags", true, true) == V{"no tags", ""});

    // While streaming, text that may still become a tag is held back.
    CHECK(split("  <thi", false, false) == V{"", ""});
    CHECK(split("<think>Let me", false, false) == V{"Let me", ""});
    CHECK(split("<think>Let me </th", false, false) == V{"Let me", ""});
    CHECK(split("<think>a</think>\n", false, false) == V{"a", ""});
    CHECK(split("Par", false, false) == V{"", "Par"});
}

TEST_CASE("split_reasoning only ever extends what it has let out", "[openai]") {
    // Streamed a byte at a time, each split's parts extend the previous ones
    // and lead to the final split.
    for (const auto& [text, inside] : std::vector<std::pair<std::string, bool>>{
             {"<think>\nLet me see: 2 + 2.\n</think>\n\nIt is 4.", false},
             {"  \n<think>a <b> </thin </think> c</think>", false},
             {"Just an answer, <think> and all.", false},
             {"Hmm.\n\n</think>\n\n\nFour.  ", true},
             {"<think> never closed </th", false}}) {
        CAPTURE(text, inside);
        const ReasoningSplit final = split_reasoning(text, inside, true);
        ReasoningSplit before;
        for (std::size_t n = 0; n <= text.size(); ++n) {
            const ReasoningSplit now =
                split_reasoning(std::string_view(text).substr(0, n), inside, n == text.size());
            CHECK(now.reasoning.starts_with(before.reasoning));
            CHECK(now.content.starts_with(before.content));
            CHECK(final.reasoning.starts_with(now.reasoning));
            CHECK(final.content.starts_with(now.content));
            before = now;
        }
    }
}

TEST_CASE("opens_reasoning tells whether a prompt leaves the reply thinking", "[openai]") {
    CHECK(opens_reasoning("<|im_start|>assistant\n<think>\n"));
    CHECK_FALSE(opens_reasoning("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    CHECK_FALSE(opens_reasoning("<|im_start|>assistant\n"));
    CHECK_FALSE(opens_reasoning(""));
}

TEST_CASE("Responses and stream chunks have OpenAI's shape", "[openai]") {
    const json full = completion_json("chatcmpl-1", 1700000000, "m", "Paris.", "stop", 10, 3);
    CHECK(full["object"] == "chat.completion");
    CHECK(full["choices"][0]["message"]["role"] == "assistant");
    CHECK(full["choices"][0]["message"]["content"] == "Paris.");
    CHECK(full["choices"][0]["finish_reason"] == "stop");
    CHECK(full["usage"]["total_tokens"] == 13);
    CHECK_FALSE(full["choices"][0]["message"].contains("reasoning_content"));
    const json thought =
        completion_json("chatcmpl-1", 1700000000, "m", "Paris.", "stop", 10, 3, "It is Paris.");
    CHECK(thought["choices"][0]["message"]["reasoning_content"] == "It is Paris.");
    CHECK(thought["choices"][0]["message"]["content"] == "Paris.");

    const json delta = chunk_json("chatcmpl-1", 1700000000, "m", json{{"content", "Pa"}}, nullptr);
    CHECK(delta["object"] == "chat.completion.chunk");
    CHECK(delta["choices"][0]["delta"]["content"] == "Pa");
    CHECK(delta["choices"][0]["finish_reason"].is_null());
    const json last = chunk_json("chatcmpl-1", 1700000000, "m", json::object(), "length");
    CHECK(last["choices"][0]["finish_reason"] == "length");

    CHECK(error_json("bad", "invalid_request_error")["error"]["message"] == "bad");
}
