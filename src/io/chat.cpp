#include "vkml/chat.hpp"

#include <fstream>

#include <nlohmann/json.hpp>

#include "io/gguf.hpp"
#include "io/jinja.hpp"
#include "vkml/error.hpp"

namespace vkml {

namespace {

// Special tokens are written either as a string or as {"content": ...}.
std::string token_text(const nlohmann::json& config, const char* key) {
    if (!config.contains(key)) return {};
    const auto& t = config[key];
    if (t.is_string()) return t.get<std::string>();
    if (t.is_object() && t.contains("content")) return t["content"].get<std::string>();
    return {};
}

}  // namespace

ChatTemplate ChatTemplate::load(const std::filesystem::path& dir) {
    const auto path = dir / "tokenizer_config.json";
    std::ifstream in(path, std::ios::binary);
    if (!in) throw Error("chat template: cannot open " + path.string());
    nlohmann::json config;
    try {
        config = nlohmann::json::parse(in);
    } catch (const nlohmann::json::exception& e) {
        throw Error("chat template: " + path.string() + " is not valid JSON: " + e.what());
    }

    std::string source;
    const auto& t = config.value("chat_template", nlohmann::json{});
    if (t.is_string()) {
        source = t.get<std::string>();
    } else if (t.is_array()) {
        for (const auto& named : t) {
            if (named.value("name", "") == "default") source = named.value("template", "");
        }
    }
    if (source.empty()) {
        throw Error("chat template: " + path.string() +
                    " has no chat_template; this model may not be tuned for chat");
    }
    return ChatTemplate{source, token_text(config, "bos_token"), token_text(config, "eos_token")};
}

ChatTemplate ChatTemplate::from_gguf(const std::filesystem::path& file) {
    const detail::Gguf gguf{file};
    const auto& m = gguf.metadata();
    if (!m.contains("tokenizer.chat_template")) {
        throw Error("chat template: " + file.string() +
                    " has no tokenizer.chat_template; this model may not be tuned for chat");
    }
    const auto token = [&](const char* key) {
        const auto& tokens = m.value("tokenizer.ggml.tokens", nlohmann::json::array());
        if (!m.contains(key)) return std::string();
        const auto id = m.at(key).get<std::size_t>();
        return id < tokens.size() ? tokens[id].get<std::string>() : std::string();
    };
    return ChatTemplate{m.at("tokenizer.chat_template").get<std::string>(),
                        token("tokenizer.ggml.bos_token_id"), token("tokenizer.ggml.eos_token_id")};
}

ChatTemplate::ChatTemplate(std::string_view source, std::string bos_token, std::string eos_token)
    : template_(std::make_unique<detail::jinja::Template>(source)),
      bos_token_(std::move(bos_token)),
      eos_token_(std::move(eos_token)) {}

ChatTemplate::~ChatTemplate() = default;
ChatTemplate::ChatTemplate(ChatTemplate&&) noexcept = default;
ChatTemplate& ChatTemplate::operator=(ChatTemplate&&) noexcept = default;

std::string ChatTemplate::render(std::span<const ChatMessage> messages,
                                 bool add_generation_prompt) const {
    nlohmann::ordered_json list = nlohmann::ordered_json::array();
    for (const ChatMessage& m : messages)
        list.push_back({{"role", m.role}, {"content", m.content}});
    const nlohmann::ordered_json context{{"messages", list},
                                         {"bos_token", bos_token_},
                                         {"eos_token", eos_token_},
                                         {"add_generation_prompt", add_generation_prompt}};
    return template_->render(context);
}

}  // namespace vkml
