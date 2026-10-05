#pragma once

// A chat model, loaded from an HF checkpoint directory or a GGUF file, and
// the loop that generates a reply with it; for vkml-chat and vkml-server.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <vkml/vkml.hpp>

namespace vkml_apps {

// The length of the longest prefix of text that does not end inside a UTF-8
// sequence. Byte-level and byte-fallback tokens can split a character across
// decoding steps, so streamed text is printed only up to here.
inline std::size_t complete_utf8_prefix(const std::string& text) {
    for (std::size_t back = 1; back <= std::min<std::size_t>(3, text.size()); ++back) {
        const auto c = static_cast<unsigned char>(text[text.size() - back]);
        if ((c & 0xC0) == 0x80) continue;  // continuation byte: keep looking for the lead
        const std::size_t needed = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        return needed > back ? text.size() - back : text.size();
    }
    return text.size();
}

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
        if (!std::filesystem::exists(path)) {
            throw vkml::Error("no model at " + path.string() +
                              ": give a model directory or a .gguf file");
        }
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

    // A reply token's log probability and the likeliest tokens' (the
    // model's, before any sampling settings), its text (maybe part of a
    // character) and where that starts in the reply.
    struct TokenLogprob {
        std::string text;
        std::size_t offset;
        vkml::TokenLogprobs logprobs;
    };

    // Each of prompt's tokens with its log probability given the ones before
    // (the first has none: NaN) and top alternatives, from the logits at every
    // position, as an echoed prompt's. The model is left holding the prompt,
    // for generate to continue.
    std::vector<TokenLogprob> score(std::span<const std::int32_t> prompt, int top) {
        model.reset();
        cached.clear();
        std::vector<TokenLogprob> out;
        std::vector<std::int32_t> seen;
        const auto add = [&](std::int32_t token, vkml::TokenLogprobs logprobs) {
            const std::size_t offset = tokenizer.decode(seen).size();
            seen.push_back(token);
            const std::string text = tokenizer.decode(seen);
            out.push_back(
                {text.substr(std::min(offset, text.size())), offset, std::move(logprobs)});
        };
        add(prompt[0], {std::numeric_limits<float>::quiet_NaN(), {}});
        // Logits in chunks of about 128 MB; row r scores the token after it.
        const std::size_t vocab = std::size_t(model.config().vocab_size);
        const std::size_t chunk = std::max<std::size_t>(1, (128u << 20) / (4 * vocab));
        for (std::size_t begin = 0; begin < prompt.size(); begin += chunk) {
            const std::size_t n = std::min(chunk, prompt.size() - begin);
            const std::vector<float> logits =
                model.forward_all(prompt.subspan(begin, n)).to_vector<float>();
            for (std::size_t r = 0; r < n && begin + r + 1 < prompt.size(); ++r) {
                const std::int32_t next = prompt[begin + r + 1];
                add(next, vkml::token_logprobs({logits.data() + r * vocab, vocab}, next, top));
            }
        }
        cached.assign(prompt.begin(), prompt.end());
        return out;
    }

    // Generates a reply to prompt (which must fit in the context) of at most
    // max_tokens tokens, or until the context is full. on_text receives the
    // reply's text as it grows, whole UTF-8 characters at a time, and returns
    // false to end the reply there. The reply's tokens go in reply, and with
    // logprobs, each one's log probability with top alternatives, before
    // on_text sees its text.
    Finish generate(std::span<const std::int32_t> prompt, vkml::Sampler& sampler, int max_tokens,
                    std::vector<std::int32_t>& reply,
                    const std::function<bool(const std::string&)>& on_text,
                    std::vector<TokenLogprob>* logprobs = nullptr, int top = 0) {
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
                          on_text, logprobs, top);
        } catch (...) {
            cached.clear();  // what the model holds is unknown: start over next time
            throw;
        }
    }

private:
    // The next token from the logits of the tokens so far (cached), which go
    // in host if given. Greedy otherwise reads back the argmax alone.
    std::int32_t pick(const vkml::Tensor& logits, vkml::Sampler& sampler,
                      std::vector<float>* host) const {
        if (!host && sampler.greedy()) return vkml::argmax(logits).to_vector<std::int32_t>()[0];
        std::vector<float> values = logits.to_vector<float>();
        // The penalty counts every token so far, prompt and reply, as HF.
        const std::int32_t next = sampler.sample(values, cached);
        if (host) *host = std::move(values);
        return next;
    }

    Finish decode(vkml::Tensor logits, vkml::Sampler& sampler, int max_tokens,
                  std::vector<std::int32_t>& reply,
                  const std::function<bool(const std::string&)>& on_text,
                  std::vector<TokenLogprob>* logprobs, int top) {
        reply.clear();
        if (logprobs) logprobs->clear();
        std::size_t sent = 0, decoded = 0;
        std::vector<float> host;
        while (max_tokens < 0 || std::int64_t(reply.size()) < max_tokens) {
            if (model.position() >= model.context_length()) return Finish::length;
            const std::int32_t next = pick(logits, sampler, logprobs ? &host : nullptr);
            if (std::ranges::find(stop, next) != stop.end()) return Finish::stop;
            reply.push_back(next);
            const std::string text = tokenizer.decode(reply);
            if (logprobs) {
                decoded = std::min(decoded, text.size());
                logprobs->push_back(
                    {text.substr(decoded), decoded, vkml::token_logprobs(host, next, top)});
                decoded = text.size();
            }
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
