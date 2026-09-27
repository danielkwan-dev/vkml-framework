#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <vkml/chat.hpp>

#include "support/safetensors_writer.hpp"

using Catch::Matchers::ContainsSubstring;
using vkml::ChatMessage;
using vkml::ChatTemplate;

namespace {

std::filesystem::path model_dir(const std::string& name, const std::string& tokenizer_config) {
    const auto dir = vkml_test::temp_path(name);
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "tokenizer_config.json", std::ios::binary) << tokenizer_config;
    return dir;
}

const std::vector<ChatMessage> kConversation{
    {"user", "Hi"}, {"assistant", "Hello!"}, {"user", "Bye"}};

}  // namespace

TEST_CASE("ChatTemplate renders a conversation with the model's special tokens", "[chat]") {
    // TinyLlama's template, as its tokenizer_config.json writes it.
    const auto dir = model_dir("vkml_chat_tinyllama", R"({
        "bos_token": "<s>", "eos_token": "</s>",
        "chat_template": "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"})");
    const ChatTemplate chat = ChatTemplate::load(dir);
    CHECK(chat.render(kConversation, true) ==
          "<|user|>\nHi</s>\n<|assistant|>\nHello!</s>\n<|user|>\nBye</s>\n<|assistant|>\n");
    CHECK(chat.render(kConversation, false) ==
          "<|user|>\nHi</s>\n<|assistant|>\nHello!</s>\n<|user|>\nBye</s>\n");
}

TEST_CASE("ChatTemplate reads token objects and named template lists", "[chat]") {
    const auto dir = model_dir("vkml_chat_forms", R"({
        "bos_token": {"content": "<B>", "lstrip": false}, "eos_token": {"content": "<E>"},
        "chat_template": [
            {"name": "tool_use", "template": "tools"},
            {"name": "default", "template": "{{ bos_token }}{% for m in messages %}{{ m.role }}:{{ m.content }}{{ eos_token }}{% endfor %}"}]})");
    CHECK(ChatTemplate::load(dir).render(kConversation, false) ==
          "<B>user:Hi<E>assistant:Hello!<E>user:Bye<E>");
}

TEST_CASE("ChatTemplate passes on errors the template raises", "[chat]") {
    const ChatTemplate chat{
        "{% for m in messages %}{% if m.role == 'assistant' %}{{ raise_exception('no assistants') "
        "}}{% endif %}{% endfor %}",
        "", ""};
    REQUIRE_THROWS_WITH(chat.render(kConversation, true), ContainsSubstring("no assistants"));
}

TEST_CASE("ChatTemplate reports models without a template", "[chat]") {
    const auto dir = model_dir("vkml_chat_none", R"({"bos_token": "<s>"})");
    REQUIRE_THROWS_WITH(ChatTemplate::load(dir), ContainsSubstring("chat_template"));
    REQUIRE_THROWS_WITH(ChatTemplate::load(vkml_test::temp_path("vkml_no_such_model_dir")),
                        ContainsSubstring("tokenizer_config.json"));
}
