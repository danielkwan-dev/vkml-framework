#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <random>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vkml {

struct SamplingOptions {
    float temperature = 0.0f;  // 0: always the most likely token (greedy)
    int top_k = 0;             // keep only the k most likely tokens; 0 keeps all
    float top_p = 1.0f;        // keep the fewest tokens whose probability sums to top_p
    std::uint64_t seed = 0;    // the same seed and logits give the same tokens
    // Tokens already seen get their logit divided by this (multiplied when
    // negative), once each, as transformers' repetition_penalty; 1 is off.
    float repetition_penalty = 1.0f;
    // Keep only tokens at least min_p times as likely as the most likely; 0 is off.
    float min_p = 0.0f;
    // As OpenAI's API, in [-2, 2]: a token's logit drops by frequency_penalty
    // for every time this sampler has picked it, and by presence_penalty once
    // if it has at all; 0 is off. The prompt does not count.
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
};

// Picks the next token from a model's logits, as transformers' generate does:
// the repetition penalty applies to the tokens seen so far, and the frequency
// and presence penalties to those picked before, then (sampling)
// logits are divided by the temperature, cut to top_k, top_p and min_p, and a
// token is drawn from the softmax of what remains.
class Sampler {
public:
    explicit Sampler(SamplingOptions options);

    // previous: the tokens so far (prompt and reply), for the repetition penalty.
    std::int32_t sample(std::span<const float> logits, std::span<const std::int32_t> previous = {});

    // Whether sample() is the argmax of the logits as given (greedy, without
    // penalties), which vkml::argmax finds without reading them back.
    bool greedy() const noexcept {
        return (options_.temperature == 0.0f || options_.top_k == 1) &&
               options_.repetition_penalty == 1.0f && options_.frequency_penalty == 0.0f &&
               options_.presence_penalty == 0.0f;
    }

private:
    std::int32_t pick(std::span<const float> logits);

    SamplingOptions options_;
    std::mt19937_64 rng_;
    std::unordered_map<std::int32_t, int>
        picked_;  // times each token was picked, for the penalties
};

// A token's log probability under logits (their log-softmax), and the top
// most likely tokens with theirs, most likely first.
struct TokenLogprobs {
    float logprob;
    std::vector<std::pair<std::int32_t, float>> top;
};
TokenLogprobs token_logprobs(std::span<const float> logits, std::int32_t token, int top);

// A checkpoint's generation_config.json: the tokens that end generation and
// the sampling its authors suggest. Settings the file leaves out stay empty.
struct GenerationConfig {
    std::vector<std::int32_t> eos_token_ids;
    std::optional<bool> do_sample;
    std::optional<float> temperature, top_p, min_p, repetition_penalty;
    std::optional<int> top_k;

    static GenerationConfig from_json(const std::filesystem::path& path);

    // options with every setting the file gives; do_sample false means greedy.
    SamplingOptions apply(SamplingOptions options) const;
};

}  // namespace vkml
