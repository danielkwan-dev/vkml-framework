#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <nlohmann/json.hpp>

#include "openai.hpp"

using Catch::Matchers::ContainsSubstring;
using vkml_openai::json;
using namespace vkml_openai;

TEST_CASE("parse_chat_request reads messages and sampling settings", "[openai]") {
    const ChatRequest r = parse_chat_request(json::parse(R"({
        "model": "anything",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": [{"type": "text", "text": "Hi "},
                                         {"type": "text", "text": "there"}]}],
        "temperature": 0.2, "top_p": 0.9, "top_k": 40, "min_p": 0.05,
        "repetition_penalty": 1.1, "presence_penalty": 0.5, "frequency_penalty": -0.25,
        "seed": 7, "max_completion_tokens": 12,
        "stream": true, "stop": "\n\n"})"));
    REQUIRE(r.messages.size() == 2);
    CHECK(r.messages[0].role == "system");
    CHECK(r.messages[1].content == "Hi there");  // text parts, joined
    CHECK(r.temperature == 0.2f);
    CHECK(r.top_p == 0.9f);
    CHECK(r.top_k == 40);
    CHECK(r.min_p == 0.05f);
    CHECK(r.repetition_penalty == 1.1f);
    CHECK(r.presence_penalty == 0.5f);
    CHECK(r.frequency_penalty == -0.25f);
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

TEST_CASE("sampling_options lays a request's settings over the defaults, in range", "[openai]") {
    const auto request = [](const std::string& settings) {
        return parse_chat_request(
            json::parse(R"({"messages": [{"role": "user", "content": "x"}])" + settings + "}"));
    };
    const vkml::SamplingOptions defaults{.temperature = 0.7f, .top_p = 0.9f, .seed = 3};
    const vkml::SamplingOptions o =
        sampling_options(request(R"(, "top_p": 0.5, "presence_penalty": 1, "seed": 9)"), defaults);
    CHECK(o.temperature == 0.7f);
    CHECK(o.top_p == 0.5f);
    CHECK(o.presence_penalty == 1.0f);
    CHECK(o.seed == 9u);
    CHECK(sampling_options(request(""), defaults).seed == 3u);

    // Out of range: a bad request, before any reply begins.
    for (const std::string bad : {R"(, "temperature": -1)", R"(, "top_p": 0)",
                                  R"(, "frequency_penalty": 2.5)", R"(, "min_p": 2)"}) {
        CAPTURE(bad);
        CHECK_THROWS_AS(sampling_options(request(bad), defaults), BadRequest);
    }
}

TEST_CASE("parse_chat_request refuses settings vkml would otherwise ignore", "[openai]") {
    const auto parse = [](const std::string& settings, bool completion = false) {
        const std::string head =
            completion ? R"({"prompt": "x")" : R"({"messages": [{"role": "user", "content": "x"}])";
        return parse_chat_request(json::parse(head + settings + "}"), completion);
    };
    for (const std::string bad : {R"(, "response_format": {"type": "json_object"})",
                                  R"(, "stream_options": {"include_usage": 1})"}) {
        CAPTURE(bad);
        CHECK_THROWS_AS(parse(bad), BadRequest);
    }
    CHECK_THROWS_AS(parse(R"(, "echo": true)"), BadRequest);  // chat has no echo
    // What asks for nothing extra passes.
    CHECK_NOTHROW(parse(R"(, "logprobs": false, "top_logprobs": 0, "echo": false,
        "response_format": {"type": "text"}, "stream_options": null)"));
    CHECK_NOTHROW(parse(R"(, "logprobs": null, "echo": false)", true));

    CHECK_FALSE(parse("").include_usage);
    CHECK(parse("").logprobs == -1);
    CHECK(parse(R"(, "logprobs": null)", true).logprobs == -1);
    CHECK(parse(R"(, "stream": true, "stream_options": {"include_usage": true})").include_usage);
}

TEST_CASE("parse_chat_request reads logprobs as each API asks for them", "[openai]") {
    const auto parse = [](const std::string& settings, bool completion = false) {
        const std::string head =
            completion ? R"({"prompt": "x")" : R"({"messages": [{"role": "user", "content": "x"}])";
        return parse_chat_request(json::parse(head + settings + "}"), completion);
    };
    // Chat: logprobs true, with top_logprobs alternatives.
    CHECK(parse(R"(, "logprobs": true)").logprobs == 0);
    CHECK(parse(R"(, "logprobs": true, "top_logprobs": 5)").logprobs == 5);
    CHECK(parse(R"(, "logprobs": false, "top_logprobs": 0)").logprobs == -1);
    // Completions: logprobs is the number of alternatives.
    CHECK(parse(R"(, "logprobs": 3)", true).logprobs == 3);
    CHECK(parse(R"(, "logprobs": 0)", true).logprobs == 0);
    for (const std::string bad : {R"(, "top_logprobs": 2)", R"(, "logprobs": 2)",
                                  R"(, "logprobs": true, "top_logprobs": 21)",
                                  R"(, "logprobs": true, "top_logprobs": -1)"}) {
        CAPTURE(bad);
        CHECK_THROWS_AS(parse(bad), BadRequest);
    }
    CHECK_THROWS_AS(parse(R"(, "logprobs": 21)", true), BadRequest);
    CHECK_THROWS_AS(parse(R"(, "logprobs": true)", true), BadRequest);
}

TEST_CASE("parse_chat_request reads a completion's prompt as text or token ids", "[openai]") {
    const auto parse = [](const std::string& rest) {
        return parse_chat_request(json::parse("{" + rest + "}"), true);
    };
    CHECK(parse(R"("prompt": "Hi")").prompt == "Hi");
    CHECK(parse(R"("prompt": ["Hi"])").prompt == "Hi");  // a list of one, as clients batch
    const ChatRequest ids = parse(R"("prompt": [1, 5, 7])");
    CHECK_FALSE(ids.prompt.has_value());
    CHECK(ids.prompt_ids == std::vector<std::int32_t>{1, 5, 7});
    CHECK(parse(R"("prompt": [[1, 5, 7]])").prompt_ids == std::vector<std::int32_t>{1, 5, 7});
    for (const std::string bad :
         {R"("prompt": ["a", "b"])", R"("prompt": [[1], [2]])", R"("prompt": [])",
          R"("prompt": [1, "a"])", R"("prompt": [-1])", R"("prompt": 3)"}) {
        CAPTURE(bad);
        CHECK_THROWS_AS(parse(bad), BadRequest);
    }
    // echo, to score the prompt: then max_tokens may be 0, nothing generated.
    const ChatRequest echo = parse(R"("prompt": "Hi", "echo": true, "max_tokens": 0)");
    CHECK(echo.echo);
    CHECK(echo.max_tokens == 0);
    CHECK_FALSE(parse(R"("prompt": "Hi")").echo);
    CHECK_THROWS_AS(parse(R"("prompt": "Hi", "max_tokens": 0)"), BadRequest);
}

TEST_CASE("Log probabilities come back in each API's shape", "[openai]") {
    const std::vector<LogprobEntry> entries{{"Hi", -0.25f, 0, {{"Hi", -0.25f}, {"Hello", -1.5f}}},
                                            {"\xe4", -2.0f, 2, {}}};
    const json chat = chat_logprobs(entries);
    REQUIRE(chat["content"].size() == 2);
    CHECK(chat["content"][0]["token"] == "Hi");
    CHECK(chat["content"][0]["logprob"] == -0.25);
    CHECK(chat["content"][0]["bytes"] == json{72, 105});
    CHECK(chat["content"][0]["top_logprobs"][1]["token"] == "Hello");
    CHECK(chat["content"][0]["top_logprobs"][1]["logprob"] == -1.5);
    // Part of a UTF-8 character: named by its bytes, as OpenAI does.
    CHECK(chat["content"][1]["token"] == "bytes:\\xe4");
    CHECK(chat["content"][1]["bytes"] == json{228});
    CHECK(chat["content"][1]["top_logprobs"] == json::array());
    CHECK_NOTHROW(chat.dump());

    const json text = text_logprobs(entries);
    CHECK(text["tokens"] == json{"Hi", "bytes:\\xe4"});
    CHECK(text["token_logprobs"] == json{-0.25, -2.0});
    CHECK(text["top_logprobs"][0] == json{{"Hi", -0.25}, {"Hello", -1.5}});
    CHECK(text["top_logprobs"][1] == json::object());
    CHECK(text["text_offset"] == json{0, 2});
    CHECK_NOTHROW(text.dump());

    // An echoed prompt's first token has nothing before it: null, as OpenAI
    // gives it.
    const std::vector<LogprobEntry> echoed{{"Hi", kNoLogprob, 0, {}},
                                           {"!", -1.0f, 2, {{"!", -1.0f}}}};
    const json e = text_logprobs(echoed);
    CHECK(e["token_logprobs"] == json{nullptr, -1.0});
    CHECK(e["top_logprobs"][0].is_null());
    CHECK(e["top_logprobs"][1] == json{{"!", -1.0}});
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

    // stream_options' include_usage: a last chunk with no choices.
    const json usage = usage_chunk_json("chatcmpl-1", 1700000000, "m", false, 10, 3);
    CHECK(usage["object"] == "chat.completion.chunk");
    CHECK(usage["choices"] == json::array());
    CHECK(usage["usage"]["total_tokens"] == 13);
    CHECK(usage_chunk_json("cmpl-1", 1700000000, "m", true, 10, 3)["object"] == "text_completion");

    CHECK(error_json("bad", "invalid_request_error")["error"]["message"] == "bad");
}

TEST_CASE("parse_chat_request passes tools and tool messages on to the template", "[openai]") {
    const ChatRequest r = parse_chat_request(json::parse(R"({
        "tools": [{"type": "function", "function": {"name": "get", "parameters": {}}}],
        "messages": [
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function",
                "function": {"name": "get", "arguments": "{\"city\": \"Oslo\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "rain"}]})"));
    REQUIRE(r.tools.is_array());
    CHECK(r.tools[0]["function"]["name"] == "get");
    REQUIRE(r.messages.size() == 3);
    CHECK(r.messages[1].content.empty());
    // Arguments as an object, as templates write them out with tojson.
    CHECK(r.messages[1].fields["tool_calls"][0]["function"]["arguments"]["city"] == "Oslo");
    CHECK(r.messages[2].fields["tool_call_id"] == "c1");
    CHECK(r.messages[0].fields.is_null());
}

TEST_CASE("parse_tool_calls takes Qwen's <tool_call> blocks out of a reply", "[openai]") {
    const ToolCalls t = parse_tool_calls(
        "Let me check.\n<tool_call>\n{\"name\": \"get\", \"arguments\": {\"city\": \"Oslo\"}}\n"
        "</tool_call>\n<tool_call>\n{\"name\": \"time\", \"arguments\": {}}\n</tool_call>",
        "call0007");
    CHECK(t.content == "Let me check.");
    REQUIRE(t.calls.size() == 2);
    // Ids of 9 letters and digits, as Mistral's template insists on: the
    // server's 8-character prefix and the call's index.
    CHECK(t.calls[0]["id"] == "call00070");
    CHECK(t.calls[1]["id"] == "call00071");
    CHECK(t.calls[0]["type"] == "function");
    CHECK(t.calls[0]["function"]["name"] == "get");
    // Arguments as a JSON string, as OpenAI's API sends them.
    CHECK(json::parse(t.calls[0]["function"]["arguments"].get<std::string>())["city"] == "Oslo");
    CHECK(t.calls[1]["function"]["name"] == "time");
    // Qwen's own Qwen2.5 GGUF files show the call in the template as
    // {{"name": ...}}, and models copy it: one pair of braces too many.
    const ToolCalls doubled = parse_tool_calls(
        "<tool_call>\n{{\"name\": \"get\", \"arguments\": {\"city\": \"Oslo\"}}}\n</tool_call>",
        "c");
    REQUIRE(doubled.calls.size() == 1);
    CHECK(doubled.calls[0]["function"]["name"] == "get");
    CHECK(doubled.content.empty());
    // Llama 3.x's: the whole reply one JSON object, its arguments "parameters".
    const ToolCalls llama = parse_tool_calls(
        R"( {"type": "function", "name": "get", "parameters": {"city": "Oslo"}})", "c");
    REQUIRE(llama.calls.size() == 1);
    CHECK(llama.content.empty());
    CHECK(json::parse(llama.calls[0]["function"]["arguments"].get<std::string>())["city"] ==
          "Oslo");
    CHECK(parse_tool_calls(R"({"answer": 42})", "c").calls.empty());
    // Mistral's: a JSON list of calls, after [TOOL_CALLS], a special token
    // the decoded text leaves out.
    for (
        const std::string reply :
        {R"( [{"name": "get", "arguments": {"city": "Oslo"}}, {"name": "time", "arguments": {}}])",
         R"([TOOL_CALLS] [{"name": "get", "arguments": {"city": "Oslo"}}, {"name": "time", "arguments": {}}])"}) {
        CAPTURE(reply);
        const ToolCalls mistral = parse_tool_calls(reply, "c");
        REQUIRE(mistral.calls.size() == 2);
        CHECK(mistral.content.empty());
        CHECK(mistral.calls[1]["function"]["name"] == "time");
    }
    CHECK(parse_tool_calls("[1, 2, 3]", "c").calls.empty());
    // Calls at the start of a reply that runs on: the rest stays text.
    const ToolCalls more = parse_tool_calls(
        R"([{"name": "get", "arguments": {"s": "a]\"}"}}]

Checking now.)",
        "c");
    REQUIRE(more.calls.size() == 1);
    CHECK(json::parse(more.calls[0]["function"]["arguments"].get<std::string>())["s"] == "a]\"}");
    CHECK(more.content == "Checking now.");
    // Text that is not a well-formed call stays text.
    CHECK(parse_tool_calls("<tool_call>not json</tool_call>", "c").calls.empty());
    CHECK(parse_tool_calls("no calls", "c").content == "no calls");

    const json full =
        completion_json("chatcmpl-1", 1, "m", "", "tool_calls", 5, 9, std::nullopt, t.calls);
    CHECK(full["choices"][0]["message"]["tool_calls"][1]["function"]["name"] == "time");
    CHECK(full["choices"][0]["message"]["content"].is_null());
    CHECK(full["choices"][0]["finish_reason"] == "tool_calls");
}

TEST_CASE("A /v1/completions request has a prompt for messages, and its own shape", "[openai]") {
    const ChatRequest r =
        parse_chat_request(json::parse(R"({"prompt": "Once upon", "max_tokens": 5})"), true);
    CHECK(r.prompt == "Once upon");
    CHECK(r.messages.empty());
    CHECK(r.max_tokens == 5);
    REQUIRE_THROWS_WITH(parse_chat_request(json::parse(R"({"prompt": 3})"), true),
                        ContainsSubstring("prompt"));
    REQUIRE_THROWS_WITH(parse_chat_request(json::parse(R"({"messages": []})"), true),
                        ContainsSubstring("prompt"));

    const json full =
        text_json("cmpl-1", 1700000000, "m", " a time", "length", json{{"total_tokens", 5}});
    CHECK(full["object"] == "text_completion");
    CHECK(full["choices"][0]["text"] == " a time");
    CHECK(full["choices"][0]["finish_reason"] == "length");
    CHECK(full["usage"]["total_tokens"] == 5);
    const json piece = text_json("cmpl-1", 1700000000, "m", " a", nullptr);
    CHECK(piece["choices"][0]["finish_reason"].is_null());
    CHECK_FALSE(piece.contains("usage"));
}
