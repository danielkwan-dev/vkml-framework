#include "vkml/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

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
    if (!(options.repetition_penalty > 0.0f)) {
        throw Error("Sampler: repetition_penalty must be more than 0, got " +
                    std::to_string(options.repetition_penalty));
    }
    if (!(options.min_p >= 0.0f && options.min_p <= 1.0f)) {
        throw Error("Sampler: min_p must be in [0, 1], got " + std::to_string(options.min_p));
    }
}

std::int32_t Sampler::sample(std::span<const float> given, std::span<const std::int32_t> previous) {
    if (given.empty()) throw Error("Sampler: the logits are empty");

    // The repetition penalty, once per distinct token seen.
    std::vector<float> penalized;
    std::span<const float> logits = given;
    if (options_.repetition_penalty != 1.0f && !previous.empty()) {
        penalized.assign(given.begin(), given.end());
        std::unordered_set<std::int32_t> seen;
        for (const std::int32_t id : previous) {
            if (id < 0 || std::size_t(id) >= penalized.size() || !seen.insert(id).second) continue;
            float& l = penalized[std::size_t(id)];
            l = l < 0.0f ? l * options_.repetition_penalty : l / options_.repetition_penalty;
        }
        logits = penalized;
    }

    const auto argmax = [&] {
        return static_cast<std::int32_t>(std::max_element(logits.begin(), logits.end()) -
                                         logits.begin());
    };
    if (options_.temperature == 0.0f || options_.top_k == 1) return argmax();

    // Candidates by descending logit, cut to top_k. Without top_k, top_p
    // needs only the most likely few in order: sorting all of Gemma's 262144
    // took 60 ms a token. Sort ever more until they reach top_p of the total.
    const double max = *std::max_element(logits.begin(), logits.end());
    const auto weight = [&](float l) { return std::exp((double(l) - max) / options_.temperature); };
    double all = 0.0;
    if (options_.top_k == 0) {
        for (const float l : logits) all += weight(l);
    }
    const bool few = options_.top_k == 0 && options_.top_p < 1.0f;
    std::vector<std::int32_t> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    const auto by_logit = [&](std::int32_t a, std::int32_t b) {
        return logits[std::size_t(a)] > logits[std::size_t(b)];
    };
    std::size_t keep = options_.top_k > 0 ? std::size_t(options_.top_k) : few ? 256 : order.size();
    for (;; keep *= 4) {
        keep = std::min(keep, order.size());
        std::partial_sort(order.begin(), order.begin() + std::ptrdiff_t(keep), order.end(),
                          by_logit);
        double sum = 0.0;
        for (std::size_t i = 0; i < keep; ++i) sum += weight(logits[std::size_t(order[i])]);
        if (!few || keep == order.size() || sum >= options_.top_p * all) break;
    }
    order.resize(keep);

    // Softmax of logits / temperature over the candidates, largest first,
    // normalized over all tokens unless top_k cut them.
    std::vector<double> p(order.size());
    for (std::size_t i = 0; i < order.size(); ++i) p[i] = weight(logits[std::size_t(order[i])]);
    double total = options_.top_k > 0 ? std::accumulate(p.begin(), p.end(), 0.0) : all;

    // top_p: the fewest candidates whose probability reaches top_p (at least one).
    if (options_.top_p < 1.0f) {
        double cumulative = 0.0;
        std::size_t n = 0;
        while (n < p.size() && cumulative < options_.top_p * total) cumulative += p[n++];
        p.resize(std::max<std::size_t>(n, 1));
        total = std::accumulate(p.begin(), p.end(), 0.0);
    }

    // min_p: candidates at least min_p times as likely as the first (p[0] is
    // the largest, so the ratio needs no normalizing).
    if (options_.min_p > 0.0f) {
        std::size_t n = 1;
        while (n < p.size() && p[n] >= options_.min_p * p[0]) ++n;
        p.resize(n);
        total = std::accumulate(p.begin(), p.end(), 0.0);
    }

    double draw = std::uniform_real_distribution<double>(0.0, total)(rng_);
    for (std::size_t i = 0; i < p.size(); ++i) {
        draw -= p[i];
        if (draw < 0.0) return order[i];
    }
    return order[p.size() - 1];  // rounding left the draw at the very end
}

GenerationConfig GenerationConfig::from_json(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw Error("GenerationConfig: cannot open " + path.string());
    GenerationConfig g;
    try {
        const nlohmann::json json = nlohmann::json::parse(in);
        if (const auto& eos = json.value("eos_token_id", nlohmann::json{}); eos.is_number()) {
            g.eos_token_ids = {eos.get<std::int32_t>()};
        } else if (eos.is_array()) {
            g.eos_token_ids = eos.get<std::vector<std::int32_t>>();
        }
        const auto read = [&](const char* key, auto& field) {
            if (const auto it = json.find(key); it != json.end() && !it->is_null()) {
                field = it->get<typename std::decay_t<decltype(field)>::value_type>();
            }
        };
        read("do_sample", g.do_sample);
        read("temperature", g.temperature);
        read("top_p", g.top_p);
        read("min_p", g.min_p);
        read("repetition_penalty", g.repetition_penalty);
        read("top_k", g.top_k);
    } catch (const nlohmann::json::exception& e) {
        throw Error("GenerationConfig: " + path.string() + ": " + e.what());
    }
    return g;
}

SamplingOptions GenerationConfig::apply(SamplingOptions options) const {
    if (temperature) options.temperature = *temperature;
    if (top_k) options.top_k = *top_k;
    if (top_p) options.top_p = *top_p;
    if (min_p) options.min_p = *min_p;
    if (repetition_penalty) options.repetition_penalty = *repetition_penalty;
    if (do_sample == false) options.temperature = 0.0f;
    return options;
}

}  // namespace vkml
