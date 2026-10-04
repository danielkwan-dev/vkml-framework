#include <cmath>
#include <cstdint>
#include <fstream>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <vkml/sampling.hpp>

#include "support/safetensors_writer.hpp"

using Catch::Matchers::ContainsSubstring;
using vkml::Sampler;
using vkml::SamplingOptions;

namespace {

// Logits for probabilities 0.5, 0.3, 0.15, 0.05 (log-probabilities work as logits).
const std::vector<float> kLogits{std::log(0.5f), std::log(0.3f), std::log(0.15f), std::log(0.05f)};

std::vector<double> frequencies(Sampler& sampler, const std::vector<float>& logits, int draws) {
    std::vector<double> counts(logits.size(), 0.0);
    for (int i = 0; i < draws; ++i) counts[std::size_t(sampler.sample(logits))] += 1.0;
    for (double& c : counts) c /= draws;
    return counts;
}

}  // namespace

TEST_CASE("Temperature zero picks the most likely token, the first of ties", "[sampling]") {
    Sampler greedy{SamplingOptions{}};
    CHECK(greedy.sample(kLogits) == 0);
    CHECK(greedy.sample(std::vector<float>{1, 5, 5, 2}) == 1);
}

TEST_CASE("A sampler is greedy without temperature or penalty, or with top_k 1", "[sampling]") {
    CHECK(Sampler{SamplingOptions{}}.greedy());
    CHECK(Sampler{SamplingOptions{.temperature = 0.7f, .top_k = 1}}.greedy());
    CHECK_FALSE(Sampler{SamplingOptions{.temperature = 0.7f}}.greedy());
    CHECK_FALSE(Sampler{SamplingOptions{.repetition_penalty = 1.1f}}.greedy());
}

TEST_CASE("Sampling follows the softmax of logits over temperature", "[sampling]") {
    Sampler sampler{SamplingOptions{.temperature = 1.0f, .seed = 42}};
    const std::vector<double> f = frequencies(sampler, kLogits, 40000);
    // 40000 draws: the standard error is at most 0.0025, so 0.01 is 4 sigma.
    CHECK(std::abs(f[0] - 0.50) < 0.01);
    CHECK(std::abs(f[1] - 0.30) < 0.01);
    CHECK(std::abs(f[2] - 0.15) < 0.01);
    CHECK(std::abs(f[3] - 0.05) < 0.01);

    // Temperature 2 flattens: probabilities go as sqrt(p), normalized.
    Sampler warm{SamplingOptions{.temperature = 2.0f, .seed = 7}};
    const std::vector<double> w = frequencies(warm, kLogits, 40000);
    const double z = std::sqrt(0.5) + std::sqrt(0.3) + std::sqrt(0.15) + std::sqrt(0.05);
    CHECK(std::abs(w[0] - std::sqrt(0.5) / z) < 0.01);
    CHECK(std::abs(w[3] - std::sqrt(0.05) / z) < 0.01);
}

TEST_CASE("top_k and top_p cut the unlikely tail before sampling", "[sampling]") {
    Sampler top2{SamplingOptions{.temperature = 1.0f, .top_k = 2, .seed = 1}};
    const std::vector<double> k = frequencies(top2, kLogits, 20000);
    CHECK(k[2] == 0.0);
    CHECK(k[3] == 0.0);
    CHECK(std::abs(k[0] - 0.5 / 0.8) < 0.015);  // renormalized over the two kept

    // top_p 0.7: 0.5 alone is short of 0.7, 0.5 + 0.3 reaches it, so two stay.
    Sampler nucleus{SamplingOptions{.temperature = 1.0f, .top_p = 0.7f, .seed = 2}};
    const std::vector<double> p = frequencies(nucleus, kLogits, 20000);
    CHECK(p[2] == 0.0);
    CHECK(p[3] == 0.0);
    CHECK(p[1] > 0.3);

    // 2000 nearly equally likely tokens, top_p 0.5: about the first 950
    // stay, more than the few the sampler sorts first.
    std::vector<float> flat(2000);
    for (std::size_t i = 0; i < flat.size(); ++i) flat[i] = -1e-4f * float(i);
    Sampler wide{SamplingOptions{.temperature = 1.0f, .top_p = 0.5f, .seed = 4}};
    const std::vector<double> w = frequencies(wide, flat, 20000);
    CHECK(std::count_if(w.begin(), w.end(), [](double f) { return f > 0.0; }) > 900);
    CHECK(std::all_of(w.begin() + 1001, w.end(), [](double f) { return f == 0.0; }));

    Sampler top1{SamplingOptions{.temperature = 5.0f, .top_k = 1, .seed = 3}};
    CHECK(frequencies(top1, kLogits, 100)[0] == 1.0);
}

TEST_CASE("The same seed gives the same draws", "[sampling]") {
    const SamplingOptions options{.temperature = 1.0f, .seed = 1234};
    Sampler a{options}, b{options};
    for (int i = 0; i < 100; ++i) CHECK(a.sample(kLogits) == b.sample(kLogits));
}

TEST_CASE("repetition_penalty lowers the logits of tokens already seen", "[sampling]") {
    // As transformers: positive logits are divided by the penalty and negative
    // ones multiplied, once per distinct token, before anything else, so the
    // most likely token under greedy decoding changes too.
    Sampler greedy{SamplingOptions{.repetition_penalty = 2.0f}};
    const std::vector<float> positive{2.0f, 1.5f, -1.0f};
    CHECK(greedy.sample(positive) == 0);
    CHECK(greedy.sample(positive, std::vector<std::int32_t>{0}) == 1);  // 2 / 2 < 1.5
    const std::vector<float> negative{-1.0f, -1.5f};
    CHECK(greedy.sample(negative, std::vector<std::int32_t>{0}) == 1);  // -1 * 2 < -1.5
    // A token seen three times is penalized once: 3 / 1.2 = 2.5 > 2.
    Sampler mild{SamplingOptions{.repetition_penalty = 1.2f}};
    CHECK(mild.sample(std::vector<float>{3.0f, 2.0f}, std::vector<std::int32_t>{0, 0, 0}) == 0);
    // Out-of-range ids in the history are ignored.
    CHECK(greedy.sample(positive, std::vector<std::int32_t>{-1, 7}) == 0);
}

TEST_CASE("frequency and presence penalties lower the tokens a sampler has picked", "[sampling]") {
    // As OpenAI's API: each logit drops by frequency_penalty for every time
    // the token was picked, and by presence_penalty once if it was at all.
    // The prompt (previous) does not count.
    const std::vector<float> logits{2.0f, 1.5f, 1.0f};
    Sampler frequency{SamplingOptions{.frequency_penalty = 0.3f}};
    CHECK(frequency.sample(logits, std::vector<std::int32_t>{0, 0, 0}) == 0);  // 2
    CHECK(frequency.sample(logits) == 0);                                      // 1.7 > 1.5
    CHECK(frequency.sample(logits) == 1);                                      // 1.4 < 1.5
    CHECK(frequency.sample(logits) == 0);                                      // 1.4 > 1.2
    CHECK(frequency.sample(logits) == 1);                                      // 1.1 < 1.2

    Sampler presence{SamplingOptions{.presence_penalty = 0.4f}};
    const std::vector<float> close{2.0f, 1.8f, 1.0f};
    CHECK(presence.sample(close) == 0);
    CHECK(presence.sample(close) == 1);  // 1.6 < 1.8
    CHECK(presence.sample(close) == 0);  // 1.6 > 1.4
    CHECK(presence.sample(close) == 0);  // still 1.6: once, however often picked

    // A negative penalty favours the tokens picked before.
    Sampler again{SamplingOptions{.presence_penalty = -1.0f}};
    CHECK(again.sample(std::vector<float>{1.0f, 1.5f}) == 1);
    CHECK(again.sample(std::vector<float>{2.0f, 1.5f}) == 1);  // 2.5 > 2

    CHECK_FALSE(Sampler{SamplingOptions{.frequency_penalty = 0.1f}}.greedy());
    CHECK_FALSE(Sampler{SamplingOptions{.presence_penalty = 0.1f}}.greedy());
}

TEST_CASE("min_p drops tokens far less likely than the most likely one", "[sampling]") {
    // Probabilities 0.5, 0.3, 0.15, 0.05: min_p 0.2 keeps those of at least
    // 0.1, and the rest are drawn in proportion.
    Sampler sampler{SamplingOptions{.temperature = 1.0f, .seed = 11, .min_p = 0.2f}};
    const std::vector<double> f = frequencies(sampler, kLogits, 20000);
    CHECK(f[3] == 0.0);
    CHECK(std::abs(f[0] - 0.5 / 0.95) < 0.02);
    CHECK(std::abs(f[2] - 0.15 / 0.95) < 0.02);
}

TEST_CASE("GenerationConfig reads generation_config.json and applies what it sets", "[sampling]") {
    const auto path = vkml_test::temp_path("vkml_generation_config.json");
    std::ofstream(path) << R"({"bos_token_id": 151643, "do_sample": true,
        "eos_token_id": [151645, 151643], "repetition_penalty": 1.1, "temperature": 0.7,
        "top_p": 0.8, "top_k": 20})";
    const vkml::GenerationConfig g = vkml::GenerationConfig::from_json(path);
    CHECK(g.eos_token_ids == std::vector<std::int32_t>{151645, 151643});
    const SamplingOptions o = g.apply(SamplingOptions{.temperature = 1.0f, .seed = 5});
    CHECK(o.temperature == 0.7f);
    CHECK(o.top_k == 20);
    CHECK(o.top_p == 0.8f);
    CHECK(o.repetition_penalty == 1.1f);
    CHECK(o.min_p == 0.0f);  // not in the file: kept
    CHECK(o.seed == 5);

    // Only eos_token_id, as most checkpoints: the options stay as they were.
    std::ofstream(path) << R"({"bos_token_id": 1, "eos_token_id": 2})";
    const vkml::GenerationConfig plain = vkml::GenerationConfig::from_json(path);
    CHECK(plain.eos_token_ids == std::vector<std::int32_t>{2});
    CHECK(plain.apply(SamplingOptions{.temperature = 0.6f}).temperature == 0.6f);

    // do_sample false asks for greedy decoding.
    std::ofstream(path) << R"({"do_sample": false, "temperature": 0.9})";
    CHECK(vkml::GenerationConfig::from_json(path).apply(SamplingOptions{}).temperature == 0.0f);

    REQUIRE_THROWS_WITH(vkml::GenerationConfig::from_json(path.string() + ".missing"),
                        ContainsSubstring("missing"));
}

TEST_CASE("Sampler rejects options outside their ranges", "[sampling]") {
    REQUIRE_THROWS_WITH(Sampler{SamplingOptions{.temperature = -1.0f}},
                        ContainsSubstring("temperature"));
    REQUIRE_THROWS_WITH(Sampler{SamplingOptions{.top_p = 0.0f}}, ContainsSubstring("top_p"));
    REQUIRE_THROWS_WITH(Sampler{SamplingOptions{.top_p = 1.5f}}, ContainsSubstring("top_p"));
    REQUIRE_THROWS_WITH(Sampler{SamplingOptions{.top_k = -2}}, ContainsSubstring("top_k"));
    REQUIRE_THROWS_WITH(Sampler{SamplingOptions{.repetition_penalty = 0.0f}},
                        ContainsSubstring("repetition_penalty"));
    REQUIRE_THROWS_WITH(Sampler{SamplingOptions{.min_p = 1.5f}}, ContainsSubstring("min_p"));
    REQUIRE_THROWS_WITH(Sampler{SamplingOptions{.frequency_penalty = 2.5f}},
                        ContainsSubstring("frequency_penalty"));
    REQUIRE_THROWS_WITH(Sampler{SamplingOptions{.presence_penalty = -3.0f}},
                        ContainsSubstring("presence_penalty"));
    Sampler s{SamplingOptions{}};
    REQUIRE_THROWS_WITH(s.sample(std::vector<float>{}), ContainsSubstring("empty"));
}
