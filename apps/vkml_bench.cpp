// Times the matmuls a LLaMA forward pass spends its time in: a weight matrix
// [n, k] applied to m rows of activations, as matmul_transposed. m = 1 is one
// decoding step, which reads every weight once per token and so is bound by
// memory bandwidth; larger m is prefill, which is bound by arithmetic.
//
//   vkml-bench [--device <name substring>]

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <span>
#include <string_view>
#include <vector>

#include <vkml/vkml.hpp>

namespace {

using Clock = std::chrono::steady_clock;

struct Case {
    const char* name;
    std::int64_t m, k, n;
    bool bf16 = false;  // weights in bf16, as checkpoints ship them
};

// TinyLlama 1.1B shapes; LLaMA 7B's are the same with every size doubled or so.
constexpr Case kCases[] = {
    {"tiny", 1, 16, 16},
    {"decode q_proj", 1, 2048, 2048},
    {"decode gate/up", 1, 2048, 5632},
    {"decode down", 1, 5632, 2048},
    {"decode lm_head", 1, 2048, 32000},
    {"m=4 gate/up", 4, 2048, 5632},
    {"m=8 gate/up", 8, 2048, 5632},
    {"m=16 gate/up", 16, 2048, 5632},
    {"m=32 gate/up", 32, 2048, 5632},
    {"m=64 gate/up", 64, 2048, 5632},
    {"prefill q_proj", 128, 2048, 2048},
    {"prefill gate/up", 128, 2048, 5632},
    {"bf16 q_proj", 1, 2048, 2048, true},
    {"bf16 k/v_proj", 1, 2048, 256, true},
    {"bf16 gate/up", 1, 2048, 5632, true},
    {"bf16 down", 1, 5632, 2048, true},
    {"bf16 lm_head", 1, 2048, 32000, true},
    {"bf16 m=2 gate/up", 2, 2048, 5632, true},
    {"bf16 m=3 gate/up", 3, 2048, 5632, true},
    {"bf16 m=4 gate/up", 4, 2048, 5632, true},
    {"bf16 m=6 gate/up", 6, 2048, 5632, true},
    {"bf16 m=8 gate/up", 8, 2048, 5632, true},
    {"bf16 prefill g/u", 128, 2048, 5632, true},
};

}  // namespace

int main(int argc, char** argv) {
    vkml::ContextOptions options;
    options.enable_validation = false;  // it would dominate the timings
    if (argc == 3 && std::string_view(argv[1]) == "--device") {
        options.device_name = argv[2];
    } else if (argc != 1) {
        std::fprintf(stderr, "usage: %s [--device <name substring>]\n", argv[0]);
        return 2;
    }

    try {
        vkml::Context context{options};
        std::printf("device  %s\n\n", context.device_info().name.c_str());
        std::printf("%-18s %6s %6s %6s %10s %10s %10s\n", "case", "m", "k", "n", "ms", "GB/s",
                    "GFLOP/s");

        float scale = 0.0f;
        for (const Case& c : kCases) {
            // A different value per case, so memory recycled from an earlier
            // case can never hold this case's answer.
            scale += 0.5f;
            const std::vector<float> a_host(static_cast<std::size_t>(c.m * c.k), scale);
            const std::vector<float> w_host(static_cast<std::size_t>(c.n * c.k), 0.25f);
            // 0x3E80 is 0.25 in bf16, so both kinds of case expect the same results.
            const std::vector<std::uint16_t> w_bf16(static_cast<std::size_t>(c.n * c.k), 0x3E80);
            const vkml::Tensor a = vkml::Tensor::from_data<float>(context, a_host, {c.m, c.k});
            // Distinct copies of the weights, cycled through, so no run finds
            // them in a cache: a model reads each weight once per token, and
            // integrated GPUs share a last-level cache big enough to hold a
            // whole small matrix, which would overstate bandwidth.
            const double weight_bytes = (c.bf16 ? 2.0 : 4.0) * double(c.n * c.k);
            const int copies = std::clamp(int(256e6 / weight_bytes), 1, 32);
            std::vector<vkml::Tensor> weights;
            for (int i = 0; i < copies; ++i) {
                weights.push_back(
                    c.bf16 ? vkml::Tensor::from_bytes(context, std::as_bytes(std::span{w_bf16}),
                                                      {c.n, c.k}, vkml::DType::BF16)
                           : vkml::Tensor::from_data<float>(context, w_host, {c.n, c.k}));
            }
            const vkml::Tensor& w = weights.front();
            // Warm up (pipeline creation, caches) and check the result: every
            // output is exactly scale / 4 * k, so a kernel that skips work
            // cannot pass for a fast one.
            const std::vector<float> check = vkml::matmul_transposed(a, w).to_vector<float>();
            const float expected = scale * 0.25f * static_cast<float>(c.k);
            if (std::ranges::any_of(check, [&](float v) { return v != expected; })) {
                std::fprintf(stderr, "%s: wrong result, expected every output to be %g\n", c.name,
                             double(expected));
                return 1;
            }

            // Batches of back-to-back runs amortize the submit and readback;
            // the best batch discounts GPU clocks still ramping up.
            constexpr int kBatches = 7;
            constexpr int kRuns = 20;
            double seconds = 1e30;
            for (int batch = 0; batch < kBatches; ++batch) {
                const auto start = Clock::now();
                std::vector<vkml::Tensor> outs;
                for (int i = 0; i < kRuns; ++i) {
                    outs.push_back(vkml::matmul_transposed(a, weights[std::size_t(i % copies)]));
                }
                (void)outs.back().to_bytes();
                seconds = std::min(
                    seconds, std::chrono::duration<double>(Clock::now() - start).count() / kRuns);
            }

            const double bytes = 4.0 * double(c.m * c.k + c.m * c.n) + weight_bytes;
            const double flops = 2.0 * double(c.m) * double(c.k) * double(c.n);
            std::printf("%-18s %6lld %6lld %6lld %10.3f %10.1f %10.1f\n", c.name,
                        static_cast<long long>(c.m), static_cast<long long>(c.k),
                        static_cast<long long>(c.n), seconds * 1e3, bytes / seconds / 1e9,
                        flops / seconds / 1e9);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
