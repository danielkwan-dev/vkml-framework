#pragma once

#include <cstdint>
#include <random>
#include <span>

namespace vkml {

struct SamplingOptions {
    float temperature = 0.0f;  // 0: always the most likely token (greedy)
    int top_k = 0;             // keep only the k most likely tokens; 0 keeps all
    float top_p = 1.0f;        // keep the fewest tokens whose probability sums to top_p
    std::uint64_t seed = 0;    // the same seed and logits give the same tokens
};

// Picks the next token from a model's logits, as transformers' generate does
// with do_sample: logits are divided by the temperature, cut to top_k and then
// top_p, and a token is drawn from the softmax of what remains.
class Sampler {
public:
    explicit Sampler(SamplingOptions options);

    std::int32_t sample(std::span<const float> logits);

private:
    SamplingOptions options_;
    std::mt19937_64 rng_;
};

}  // namespace vkml
