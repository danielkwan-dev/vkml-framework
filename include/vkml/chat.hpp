#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace vkml {

namespace detail::jinja {
class Template;
}  // namespace detail::jinja

struct ChatMessage {
    std::string role;  // "system", "user" or "assistant"
    std::string content;
};

// Turns a conversation into the prompt text a chat model was trained on, by
// running the Jinja template the model ships in tokenizer_config.json, as
// transformers' apply_chat_template does. Templates using Jinja beyond what
// chat templates need are rejected when loading or rendering, not rendered
// approximately.
class ChatTemplate {
public:
    // Reads chat_template (a string, or a list of named templates of which
    // "default" is used), bos_token and eos_token from dir/tokenizer_config.json.
    static ChatTemplate load(const std::filesystem::path& dir);

    // Reads tokenizer.chat_template, and the BOS and EOS tokens by their ids
    // in tokenizer.ggml.tokens, from a GGUF file's metadata.
    static ChatTemplate from_gguf(const std::filesystem::path& file);

    ChatTemplate(std::string_view source, std::string bos_token, std::string eos_token);
    ~ChatTemplate();
    ChatTemplate(ChatTemplate&&) noexcept;
    ChatTemplate& operator=(ChatTemplate&&) noexcept;

    // With add_generation_prompt, ends with the text that starts the
    // assistant's reply. enable_thinking, if given, is passed to the template
    // as apply_chat_template's keyword argument of that name is: false asks
    // reasoning models (Qwen3) to answer without thinking first; templates
    // without it ignore it. Throws vkml::Error if the template raises an
    // error, for example for roles it does not accept.
    std::string render(std::span<const ChatMessage> messages, bool add_generation_prompt,
                       std::optional<bool> enable_thinking = std::nullopt) const;

    const std::string& bos_token() const noexcept { return bos_token_; }
    const std::string& eos_token() const noexcept { return eos_token_; }

private:
    std::unique_ptr<detail::jinja::Template> template_;
    std::string bos_token_;
    std::string eos_token_;
};

}  // namespace vkml
