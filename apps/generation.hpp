#pragma once

// A chat model, loaded from an HF checkpoint directory or a GGUF file, and
// the loop that generates a reply with it; for vkml-chat and vkml-server.

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <vkml/vkml.hpp>

#include "text_stream.hpp"

namespace vkml_apps {

struct ChatModel {
    vkml::ChatTemplate chat;
    vkml::Tokenizer tokenizer;
    vkml::Llama model;
    std::optional<vkml::GenerationConfig> generation;  // generation_config.json, if any
    // The tokens that end a reply: the config's, the generation config's and
    // the template's end-of-sequence tokens.
    std::vector<std::int32_t> stop;
    // The tokens the model has processed, in order, so a request that begins
    // with them (the next turn of a conversation) processes only the rest.
    std::vector<std::int32_t> cached;

    // template_dir, if given, is a model directory whose chat template to use
    // instead of the model's own (a GGUF file's may be older, without tools).
    static ChatModel load(vkml::Context& context, const std::filesystem::path& path,
                          std::int64_t context_length, const vkml::LlamaOptions& options,
                          const std::filesystem::path& template_dir = {}) {
        const bool gguf = path.extension() == ".gguf";
        auto chat = !template_dir.empty() ? vkml::ChatTemplate::load(template_dir)
                    : gguf                ? vkml::ChatTemplate::from_gguf(path)
                                          : vkml::ChatTemplate::load(path);
        auto tokenizer =
            gguf ? vkml::Tokenizer::from_gguf(path) : vkml::Tokenizer{path / "tokenizer.json"};
        const auto config = gguf ? vkml::LlamaConfig::from_gguf(path)
                                 : vkml::LlamaConfig::from_json(path / "config.json");
        auto model = vkml::Llama::load(context, path,
                                       std::min(context_length, config.max_positions), options);
        std::optional<vkml::GenerationConfig> generation;
        if (!gguf && std::filesystem::exists(path / "generation_config.json")) {
            generation = vkml::GenerationConfig::from_json(path / "generation_config.json");
        }
        std::vector<std::int32_t> stop = model.config().eos_token_ids;
        if (generation) {
            stop.insert(stop.end(), generation->eos_token_ids.begin(),
                        generation->eos_token_ids.end());
        }
        if (const auto id = tokenizer.token_id(chat.eos_token())) stop.push_back(*id);
        return ChatModel{std::move(chat),       std::move(tokenizer), std::move(model),
                         std::move(generation), std::move(stop),      {}};
    }

    // The prompt for messages, as the model's template writes it, with
    // enable_thinking passed to it if given.
    std::vector<std::int32_t> encode(const std::vector<vkml::ChatMessage>& messages,
                                     std::optional<bool> enable_thinking = std::nullopt) const {
        // HF's apply_chat_template adds no special tokens of its own: the
        // template writes the ones the model expects.
        return tokenizer.encode(chat.render(messages, true, enable_thinking), false);
    }

    enum class Finish { stop, length };

    // Generates a reply to prompt (which must fit in the context) of at most
    // max_tokens tokens, or until the context is full. on_text receives the
    // reply's text as it grows, whole UTF-8 characters at a time, and returns
    // false to end the reply there. The reply's tokens go in reply.
    Finish generate(std::span<const std::int32_t> prompt, vkml::Sampler& sampler, int max_tokens,
                    std::vector<std::int32_t>& reply,
                    const std::function<bool(const std::string&)>& on_text) {
        // Only tokens after the part the cache already holds are processed.
        std::size_t common =
            std::size_t(std::ranges::mismatch(cached, prompt).in2 - prompt.begin());
        common = std::min(common, prompt.size() - 1);  // forward at least one token for logits
        // The model may start over instead, its sliding windows' caches
        // having moved past common.
        common = std::size_t(model.rewind(std::int64_t(common)));
        cached.assign(prompt.begin(), prompt.end());
        try {
            return decode(model.forward(prompt.subspan(common)), sampler, max_tokens, reply,
                          on_text);
        } catch (...) {
            cached.clear();  // what the model holds is unknown: start over next time
            throw;
        }
    }

private:
    // The next token from the logits of the tokens so far (cached). Greedy
    // reads back the argmax alone, not every logit.
    std::int32_t pick(const vkml::Tensor& logits, vkml::Sampler& sampler) const {
        if (sampler.greedy()) return vkml::argmax(logits).to_vector<std::int32_t>()[0];
        // The penalty counts every token so far, prompt and reply, as HF.
        return sampler.sample(logits.to_vector<float>(), cached);
    }

    Finish decode(vkml::Tensor logits, vkml::Sampler& sampler, int max_tokens,
                  std::vector<std::int32_t>& reply,
                  const std::function<bool(const std::string&)>& on_text) {
        reply.clear();
        std::size_t sent = 0;
        while (max_tokens < 0 || std::int64_t(reply.size()) < max_tokens) {
            if (model.position() >= model.context_length()) return Finish::length;
            const std::int32_t next = pick(logits, sampler);
            if (std::ranges::find(stop, next) != stop.end()) return Finish::stop;
            reply.push_back(next);
            const std::string text = tokenizer.decode(reply);
            const std::size_t ready = complete_utf8_prefix(text);
            if (ready > sent) {
                sent = ready;
                if (!on_text(text.substr(0, ready))) return Finish::stop;
            }
            logits = model.forward({&next, 1});
            cached.push_back(next);
        }
        return Finish::length;
    }
};

}  // namespace vkml_apps
