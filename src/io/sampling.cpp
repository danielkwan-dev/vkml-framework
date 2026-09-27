#include "vkml/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include "vkml/error.hpp"

namespace vkml {

Sampler::Sampler(SamplingOptions options) : options_(options), rng_(options.seed) {
    if (!(options.temperature >= 0.0f)) {
        throw Error("Sampler: temperature must be 0 or more, got " +
                    std::to_string(options.temperature));
    }
    if (!(options.top_p > 0.0f && options.top_p <= 1.0f)) {
        throw Error("Sampler: top_p must be in (0, 1], got " + std::to_string(options.top_p));
    }
    if (options.top_k < 0) {
        throw Error("Sampler: top_k must be 0 (off) or more, got " + std::to_string(options.top_k));
    }
}

std::int32_t Sampler::sample(std::span<const float> logits) {
    if (logits.empty()) throw Error("Sampler: the logits are empty");
    const auto argmax = [&] {
        return static_cast<std::int32_t>(std::max_element(logits.begin(), logits.end()) -
                                         logits.begin());
    };
    if (options_.temperature == 0.0f || options_.top_k == 1) return argmax();

    // Candidates by descending logit, cut to top_k.
    std::vector<std::int32_t> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    const std::size_t keep = options_.top_k > 0
                                 ? std::min<std::size_t>(std::size_t(options_.top_k), order.size())
                                 : order.size();
    const auto by_logit = [&](std::int32_t a, std::int32_t b) {
        return logits[std::size_t(a)] > logits[std::size_t(b)];
    };
    std::partial_sort(order.begin(), order.begin() + std::ptrdiff_t(keep), order.end(), by_logit);
    order.resize(keep);

    // Softmax of logits / temperature over the candidates, largest first.
    const double max = logits[std::size_t(order.front())];
    std::vector<double> p(order.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        p[i] = std::exp((double(logits[std::size_t(order[i])]) - max) / options_.temperature);
    }
    double total = std::accumulate(p.begin(), p.end(), 0.0);

    // top_p: the fewest candidates whose probability reaches top_p (at least one).
    if (options_.top_p < 1.0f) {
        double cumulative = 0.0;
        std::size_t n = 0;
        while (n < p.size() && cumulative < options_.top_p * total) cumulative += p[n++];
        p.resize(std::max<std::size_t>(n, 1));
        total = std::accumulate(p.begin(), p.end(), 0.0);
    }

    double draw = std::uniform_real_distribution<double>(0.0, total)(rng_);
    for (std::size_t i = 0; i < p.size(); ++i) {
        draw -= p[i];
        if (draw < 0.0) return order[i];
    }
    return order[p.size() - 1];  // rounding left the draw at the very end
}

}  // namespace vkml
