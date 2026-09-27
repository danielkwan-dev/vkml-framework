#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <vkml/sampling.hpp>

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

    Sampler top1{SamplingOptions{.temperature = 5.0f, .top_k = 1, .seed = 3}};
    CHECK(frequencies(top1, kLogits, 100)[0] == 1.0);
}

TEST_CASE("The same seed gives the same draws", "[sampling]") {
    const SamplingOptions options{.temperature = 1.0f, .seed = 1234};
    Sampler a{options}, b{options};
    for (int i = 0; i < 100; ++i) CHECK(a.sample(kLogits) == b.sample(kLogits));
}

TEST_CASE("Sampler rejects options outside their ranges", "[sampling]") {
    REQUIRE_THROWS_WITH(Sampler{SamplingOptions{.temperature = -1.0f}},
                        ContainsSubstring("temperature"));
    REQUIRE_THROWS_WITH(Sampler{SamplingOptions{.top_p = 0.0f}}, ContainsSubstring("top_p"));
    REQUIRE_THROWS_WITH(Sampler{SamplingOptions{.top_p = 1.5f}}, ContainsSubstring("top_p"));
    REQUIRE_THROWS_WITH(Sampler{SamplingOptions{.top_k = -2}}, ContainsSubstring("top_k"));
    Sampler s{SamplingOptions{}};
    REQUIRE_THROWS_WITH(s.sample(std::vector<float>{}), ContainsSubstring("empty"));
}
