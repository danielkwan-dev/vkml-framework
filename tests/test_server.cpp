// First: it includes httplib, which must come before any system header.
#include "server.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "io/pretokenize.hpp"
#include "support/safetensors_writer.hpp"

using vkml_openai::json;

namespace {

// A byte-level tokenizer (byte b is token b, then <|im_start|> 256 and
// <|im_end|> 257, the end of a turn, then each of chunks, a whole piece of
// text as one token) and a ChatML template.
void write_tokenizer_files(const std::filesystem::path& dir,
                           const std::vector<std::string>& chunks = {}) {
    json tokenizer_vocab = json::object();
    for (int b = 0; b < 256; ++b) {
        tokenizer_vocab[vkml::detail::bytes_to_unicode(std::string(1, char(b)))] = b;
    }
    for (std::size_t i = 0; i < chunks.size(); ++i) {
        tokenizer_vocab[vkml::detail::bytes_to_unicode(chunks[i])] = 258 + i;
    }
    const json tokenizer = {
        {"added_tokens",
         {{{"id", 256}, {"content", "<|im_start|>"}, {"special", true}, {"normalized", false}},
          {{"id", 257}, {"content", "<|im_end|>"}, {"special", true}, {"normalized", false}}}},
        {"normalizer", nullptr},
        {"pre_tokenizer",
         {{"type", "ByteLevel"}, {"add_prefix_space", false}, {"use_regex", true}}},
        {"post_processor", nullptr},
        {"decoder", {{"type", "ByteLevel"}}},
        {"model", {{"type", "BPE"}, {"vocab", tokenizer_vocab}, {"merges", json::array()}}}};
    std::ofstream(dir / "tokenizer.json", std::ios::binary) << tokenizer.dump();
    const json config = {
        {"chat_template",
         "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
         "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"},
        {"eos_token", "<|im_end|>"}};
    std::ofstream(dir / "tokenizer_config.json", std::ios::binary) << config.dump();
}

// A LLaMA of random weights with write_tokenizer_files' tokenizer: its text is
// noise, but every route can run on it.
std::filesystem::path write_tiny_chat_model(const std::string& name) {
    const auto dir = vkml_test::temp_path(name);
    std::filesystem::create_directories(dir);
    constexpr std::int64_t d = 32, ff = 64, layers = 2, heads = 4, kv_heads = 2, vocab = 258;
    constexpr std::int64_t hd = d / heads;
    std::ofstream(dir / "config.json") << R"({"architectures": ["LlamaForCausalLM"],
        "model_type": "llama", "hidden_size": 32, "intermediate_size": 64,
        "num_attention_heads": 4, "num_key_value_heads": 2, "num_hidden_layers": 2,
        "vocab_size": 258, "max_position_embeddings": 512, "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0, "tie_word_embeddings": true, "hidden_act": "silu",
        "eos_token_id": 257})";

    std::mt19937 rng(7);
    std::normal_distribution<float> normal(0.0f, 0.5f);
    std::vector<vkml_test::Entry> entries;
    const auto random = [&](const std::string& n, vkml::Shape shape) {
        std::vector<float> v(std::size_t(shape[0] * (shape.size() > 1 ? shape[1] : 1)));
        for (float& x : v) x = normal(rng);
        entries.push_back({n, "F32", shape, vkml_test::raw(v)});
    };
    const auto ones = [&](const std::string& n) {
        entries.push_back({n, "F32", {d}, vkml_test::raw(std::vector<float>(d, 1.0f))});
    };
    random("model.embed_tokens.weight", {vocab, d});
    for (std::int64_t i = 0; i < layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        ones(p + "input_layernorm.weight");
        ones(p + "post_attention_layernorm.weight");
        random(p + "self_attn.q_proj.weight", {heads * hd, d});
        random(p + "self_attn.k_proj.weight", {kv_heads * hd, d});
        random(p + "self_attn.v_proj.weight", {kv_heads * hd, d});
        random(p + "self_attn.o_proj.weight", {d, heads * hd});
        random(p + "mlp.gate_proj.weight", {ff, d});
        random(p + "mlp.up_proj.weight", {ff, d});
        random(p + "mlp.down_proj.weight", {d, ff});
    }
    ones("model.norm.weight");
    vkml_test::write_safetensors(dir / "model.safetensors", entries);

    write_tokenizer_files(dir);
    return dir;
}

// A LLaMA that answers every chat prompt with chunks, then ends its turn. Its
// layers are zero, so the residual stream holds the last token's embedding
// alone; embeddings are one-hot and the output projection maps each token to
// its successor: the generation prompt's last "\n" to the first chunk, each
// chunk to the next, the last to <|im_end|>.
std::filesystem::path write_script_model(const std::string& name,
                                         const std::vector<std::string>& chunks) {
    const auto dir = vkml_test::temp_path(name);
    std::filesystem::create_directories(dir);
    const std::int64_t vocab = 258 + std::int64_t(chunks.size());
    const std::int64_t d = (vocab + 7) / 8 * 8, ff = 16, heads = 4, kv_heads = 2;
    const std::int64_t hd = d / heads;
    std::ofstream(dir / "config.json") << json{{"architectures", {"LlamaForCausalLM"}},
                                               {"model_type", "llama"},
                                               {"hidden_size", d},
                                               {"intermediate_size", ff},
                                               {"num_attention_heads", heads},
                                               {"num_key_value_heads", kv_heads},
                                               {"num_hidden_layers", 1},
                                               {"vocab_size", vocab},
                                               {"max_position_embeddings", 512},
                                               {"rms_norm_eps", 1e-5},
                                               {"rope_theta", 10000.0},
                                               {"tie_word_embeddings", false},
                                               {"hidden_act", "silu"},
                                               {"eos_token_id", 257}}
                                              .dump();
    std::vector<vkml_test::Entry> entries;
    const auto matrix = [&](const std::string& n, std::int64_t rows, std::int64_t cols,
                            const std::vector<std::pair<std::int64_t, std::int64_t>>& ones = {}) {
        std::vector<float> v(std::size_t(rows * cols), 0.0f);
        for (const auto& [r, c] : ones) v[std::size_t(r * cols + c)] = 1.0f;
        entries.push_back({n, "F32", {rows, cols}, vkml_test::raw(v)});
    };
    const auto norm = [&](const std::string& n) {
        entries.push_back(
            {n, "F32", {d}, vkml_test::raw(std::vector<float>(std::size_t(d), 1.0f))});
    };
    std::vector<std::pair<std::int64_t, std::int64_t>> one_hot, successors;
    for (std::int64_t t = 0; t < vocab; ++t) one_hot.push_back({t, t});
    successors.push_back({258, '\n'});  // [next, current]
    for (std::int64_t i = 0; i < std::int64_t(chunks.size()); ++i) {
        successors.push_back({i + 1 < std::int64_t(chunks.size()) ? 259 + i : 257, 258 + i});
    }
    matrix("model.embed_tokens.weight", vocab, d, one_hot);
    const std::string p = "model.layers.0.";
    norm(p + "input_layernorm.weight");
    norm(p + "post_attention_layernorm.weight");
    matrix(p + "self_attn.q_proj.weight", heads * hd, d);
    matrix(p + "self_attn.k_proj.weight", kv_heads * hd, d);
    matrix(p + "self_attn.v_proj.weight", kv_heads * hd, d);
    matrix(p + "self_attn.o_proj.weight", d, heads * hd);
    matrix(p + "mlp.gate_proj.weight", ff, d);
    matrix(p + "mlp.up_proj.weight", ff, d);
    matrix(p + "mlp.down_proj.weight", d, ff);
    norm("model.norm.weight");
    matrix("lm_head.weight", vocab, d, successors);
    vkml_test::write_safetensors(dir / "model.safetensors", entries);
    write_tokenizer_files(dir, chunks);
    return dir;
}

// vkml-server's routes on a free local port, until destroyed.
struct RunningServer {
    vkml::Context context;
    vkml_apps::ChatModel model;
    vkml_server::Server server;
    httplib::Server http;
    int port = 0;
    std::thread thread;

    // reasoning: as --reasoning-content.
    explicit RunningServer(const std::filesystem::path& dir, bool reasoning = false)
        : model(vkml_apps::ChatModel::load(context, dir, 256, {})),
          server(model, "tiny", std::nullopt, reasoning) {
        vkml_server::add_routes(http, server, "");
        port = http.bind_to_any_port("127.0.0.1");
        thread = std::thread([this] { http.listen_after_bind(); });
        http.wait_until_ready();
    }
    ~RunningServer() {
        http.stop();
        thread.join();
    }
};

// The data of each server-sent event, [DONE] included.
std::vector<std::string> events(const std::string& body) {
    std::vector<std::string> out;
    for (std::size_t at = 0; (at = body.find("data: ", at)) != std::string::npos;) {
        const std::size_t end = body.find("\n\n", at);
        out.push_back(body.substr(at + 6, end - at - 6));
        at = end;
    }
    return out;
}

// The text the logprobs entries' "bytes" fields spell, joined, as a
// response gives it: bytes that are not UTF-8 as U+FFFD.
std::string joined_bytes(const json& entries) {
    std::string out;
    for (const json& e : entries) {
        for (const json& b : e["bytes"]) out += char(b.get<int>());
    }
    return json::parse(json(out).dump(-1, ' ', false, json::error_handler_t::replace))
        .get<std::string>();
}

}  // namespace

TEST_CASE("vkml-server answers chat and completion requests over HTTP", "[server]") {
    RunningServer s{write_tiny_chat_model("vkml_tiny_chat_server")};
    httplib::Client client("127.0.0.1", s.port);
    client.set_read_timeout(120);
    const auto post = [&](const std::string& path, const json& body) {
        const auto r = client.Post(path, body.dump(), "application/json");
        REQUIRE(r);
        return std::pair{r->status, r->body};
    };
    const json messages = json::array({{{"role", "user"}, {"content", "Hi there"}}});

    const auto health = client.Get("/health");
    REQUIRE(health);
    CHECK(health->status == 200);
    const auto models = client.Get("/v1/models");
    REQUIRE(models);
    CHECK(json::parse(models->body)["data"][0]["id"] == "tiny");

    // A reply, then the same streamed: the same text, the same usage.
    const json request = {{"messages", messages}, {"temperature", 0}, {"max_tokens", 12}};
    const auto [status, body] = post("/v1/chat/completions", request);
    INFO(body);
    REQUIRE(status == 200);
    const json reply = json::parse(body);
    const std::string content = reply["choices"][0]["message"]["content"];
    const int completion_tokens = reply["usage"]["completion_tokens"];
    CHECK(completion_tokens >= 1);
    CHECK(completion_tokens <= 12);

    json streamed = request;
    streamed["stream"] = true;
    streamed["stream_options"] = {{"include_usage", true}};
    const std::vector<std::string> chunks = events(post("/v1/chat/completions", streamed).second);
    REQUIRE(chunks.size() >= 3);
    CHECK(chunks.back() == "[DONE]");
    std::string text;
    json usage;
    for (std::size_t i = 0; i + 1 < chunks.size(); ++i) {
        const json c = json::parse(chunks[i]);
        if (c["choices"].empty()) {
            usage = c["usage"];
            continue;
        }
        text += c["choices"][0]["delta"].value("content", "");
    }
    CHECK(text == content);
    CHECK(usage == reply["usage"]);

    // Logprobs: one entry a token, whose bytes spell the reply, the greedy
    // pick the likeliest; and the same when streamed.
    json with_logprobs = request;
    with_logprobs["logprobs"] = true;
    with_logprobs["top_logprobs"] = 3;
    const json lp = json::parse(post("/v1/chat/completions", with_logprobs).second);
    const json& entries = lp["choices"][0]["logprobs"]["content"];
    REQUIRE(entries.size() == std::size_t(completion_tokens));
    CHECK(joined_bytes(entries) == content);
    for (const json& e : entries) {
        REQUIRE(e["top_logprobs"].size() == 3);
        CHECK(e["top_logprobs"][0]["token"] == e["token"]);
    }
    with_logprobs["stream"] = true;
    json streamed_entries = json::array();
    for (const std::string& c : events(post("/v1/chat/completions", with_logprobs).second)) {
        if (c == "[DONE]") break;
        const json chunk = json::parse(c);
        if (chunk["choices"][0].contains("logprobs")) {
            for (const json& e : chunk["choices"][0]["logprobs"]["content"])
                streamed_entries.push_back(e);
        }
    }
    CHECK(streamed_entries == entries);

    // A stop string from the reply (an ASCII character after its first) cuts
    // it there.
    const std::size_t at =
        content.find_first_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", 1);
    if (at != std::string::npos) {
        json stopped = request;
        const std::string stop = content.substr(at, 1);
        stopped["stop"] = stop;
        const json r = json::parse(post("/v1/chat/completions", stopped).second);
        CHECK(r["choices"][0]["message"]["content"] == content.substr(0, content.find(stop)));
        CHECK(r["choices"][0]["finish_reason"] == "stop");
    }

    // A completion echoing its prompt: the prompt's tokens first, the first
    // without a log probability, offsets into the echoed text.
    const json echo = json::parse(post("/v1/completions", {{"prompt", "Hello"},
                                                           {"echo", true},
                                                           {"logprobs", 2},
                                                           {"max_tokens", 3},
                                                           {"temperature", 0}})
                                      .second);
    const std::string echoed = echo["choices"][0]["text"];
    CHECK(echoed.starts_with("Hello"));
    const json& text_lp = echo["choices"][0]["logprobs"];
    CHECK(text_lp["tokens"].size() == std::size_t(echo["usage"]["prompt_tokens"].get<int>() +
                                                  echo["usage"]["completion_tokens"].get<int>()));
    CHECK(text_lp["token_logprobs"][0].is_null());
    CHECK(text_lp["text_offset"][0] == 0);
    CHECK(text_lp["text_offset"][5] == 5);  // the first generated token, after "Hello"
    const json scored = json::parse(
        post("/v1/completions", {{"prompt", "Hello"}, {"echo", true}, {"max_tokens", 0}}).second);
    CHECK(scored["choices"][0]["text"] == "Hello");
    CHECK(scored["usage"]["completion_tokens"] == 0);

    // Requests it cannot serve: 400, with OpenAI's error shape.
    for (const json& bad :
         {json{{"messages", messages}, {"temperature", -1}},
          json{{"messages", messages}, {"response_format", {{"type", "json_object"}}}},
          json{{"messages", json::array()}}}) {
        CAPTURE(bad.dump());
        const auto [code, error] = post("/v1/chat/completions", bad);
        CHECK(code == 400);
        CHECK(json::parse(error)["error"]["type"] == "invalid_request_error");
    }
}

TEST_CASE("vkml-server returns the tool calls a model writes, streamed or not", "[server]") {
    // Qwen's <tool_call> block, and Llama 3's reply of one JSON object, which
    // a stream holds back until it is whole.
    const auto format = GENERATE(0, 1);
    CAPTURE(format);
    const std::vector<std::string> script =
        format == 0
            ? std::vector<std::string>{"<tool_call>\n", R"({"name": "get_weather", )",
                                       R"("arguments": {"city": "Paris"}})", "\n</tool_call>"}
            : std::vector<std::string>{R"({"name": "get_weather", )",
                                       R"("parameters": {"city": "Paris"}})"};
    std::string text;
    for (const std::string& chunk : script) text += chunk;
    RunningServer s{write_script_model("vkml_script_server_" + std::to_string(format), script)};
    httplib::Client client("127.0.0.1", s.port);
    client.set_read_timeout(120);
    const auto post = [&](const json& body) {
        const auto r = client.Post("/v1/chat/completions", body.dump(), "application/json");
        REQUIRE(r);
        REQUIRE(r->status == 200);
        return r->body;
    };
    const json tools = json::array(
        {{{"type", "function"},
          {"function", {{"name", "get_weather"}, {"parameters", {{"type", "object"}}}}}}});
    json request = {{"messages", json::array({{{"role", "user"}, {"content", "Weather?"}}})},
                    {"temperature", 0}};

    // Without tools the call is only text.
    const json plain = json::parse(post(request));
    CHECK(plain["choices"][0]["message"]["content"] == text);
    CHECK(plain["choices"][0]["finish_reason"] == "stop");

    request["tools"] = tools;
    const json reply = json::parse(post(request));
    const json& message = reply["choices"][0]["message"];
    CHECK(reply["choices"][0]["finish_reason"] == "tool_calls");
    CHECK(message["content"].is_null());
    REQUIRE(message["tool_calls"].size() == 1);
    CHECK(message["tool_calls"][0]["function"]["name"] == "get_weather");
    CHECK(json::parse(message["tool_calls"][0]["function"]["arguments"].get<std::string>()) ==
          json{{"city", "Paris"}});

    // Streamed: no piece of the call as text, the call whole, then the finish.
    request["stream"] = true;
    std::string content;
    json calls, finish;
    for (const std::string& e : events(post(request))) {
        if (e == "[DONE]") break;
        const json chunk = json::parse(e);
        const json& choice = chunk["choices"][0];
        content += choice["delta"].value("content", "");
        if (choice["delta"].contains("tool_calls")) calls = choice["delta"]["tool_calls"];
        if (!choice["finish_reason"].is_null()) finish = choice["finish_reason"];
    }
    CHECK(content.empty());
    REQUIRE(calls.size() == 1);
    CHECK(calls[0]["index"] == 0);
    CHECK(calls[0]["id"].get<std::string>().size() == 9);  // as Mistral's template insists
    CHECK(calls[0]["function"]["name"] == "get_weather");
    CHECK(finish == "tool_calls");
}

TEST_CASE("vkml-server moves a reply's think block to reasoning_content", "[server]") {
    // As --reasoning-content does: apart from the answer, streamed or not.
    RunningServer s{write_script_model("vkml_script_server_think",
                                       {"<think>\n", "Let me see.", "\n</think>\n\n", "Paris."}),
                    true};
    httplib::Client client("127.0.0.1", s.port);
    client.set_read_timeout(120);
    json request = {{"messages", json::array({{{"role", "user"}, {"content", "Capital?"}}})},
                    {"temperature", 0}};
    const auto r = client.Post("/v1/chat/completions", request.dump(), "application/json");
    REQUIRE(r);
    REQUIRE(r->status == 200);
    const json message = json::parse(r->body)["choices"][0]["message"];
    CHECK(message["reasoning_content"] == "Let me see.");
    CHECK(message["content"] == "Paris.");

    request["stream"] = true;
    const auto streamed = client.Post("/v1/chat/completions", request.dump(), "application/json");
    REQUIRE(streamed);
    std::string reasoning, content;
    for (const std::string& e : events(streamed->body)) {
        if (e == "[DONE]") break;
        const json chunk = json::parse(e);
        const json& delta = chunk["choices"][0]["delta"];
        reasoning += delta.value("reasoning_content", "");
        content += delta.value("content", "");
    }
    CHECK(reasoning == "Let me see.");
    CHECK(content == "Paris.");
}
