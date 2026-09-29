#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <random>
#include <span>
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
};

// Picks the next token from a model's logits, as transformers' generate does:
// the repetition penalty applies to the tokens seen so far, then (sampling)
// logits are divided by the temperature, cut to top_k, top_p and min_p, and a
// token is drawn from the softmax of what remains.
class Sampler {
public:
    explicit Sampler(SamplingOptions options);

    // previous: the tokens so far (prompt and reply), for the repetition penalty.
    std::int32_t sample(std::span<const float> logits, std::span<const std::int32_t> previous = {});

private:
    SamplingOptions options_;
    std::mt19937_64 rng_;
};

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
