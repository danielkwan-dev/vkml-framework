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
#include <string_view>
#include <vector>

#include <vkml/vkml.hpp>

namespace {

using Clock = std::chrono::steady_clock;

struct Case {
    const char* name;
    std::int64_t m, k, n;
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
            const vkml::Tensor a = vkml::Tensor::from_data<float>(context, a_host, {c.m, c.k});
            const vkml::Tensor w = vkml::Tensor::from_data<float>(context, w_host, {c.n, c.k});
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
                for (int i = 0; i < kRuns; ++i) outs.push_back(vkml::matmul_transposed(a, w));
                (void)outs.back().to_bytes();
                seconds = std::min(
                    seconds, std::chrono::duration<double>(Clock::now() - start).count() / kRuns);
            }

            const double bytes = 4.0 * double(c.m * c.k + c.n * c.k + c.m * c.n);
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
