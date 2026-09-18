#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <vkml_shaders/rope.spv.hpp>

#include "core/runtime.hpp"
#include "core/storage.hpp"
#include "vkml/ops.hpp"

namespace vkml {

namespace {

using detail::TensorAccess;

struct RopeParams {
    std::uint32_t pairs;
    std::uint32_t half_dim;
    std::uint32_t heads;
    std::uint32_t seq;
    std::uint32_t start_pos;
};

}  // namespace

Tensor rope_table(Context& context, std::int64_t max_positions, std::int64_t head_dim,
                  float theta) {
    if (head_dim <= 0 || head_dim % 2 != 0 || max_positions < 0) {
        throw Error(
            "rope_table: head_dim must be positive and even and max_positions "
            "non-negative, got head_dim " +
            std::to_string(head_dim) + ", max_positions " + std::to_string(max_positions));
    }
    const auto positions = static_cast<std::size_t>(max_positions);
    const auto half = static_cast<std::size_t>(head_dim / 2);
    std::vector<double> inv_freq(half);
    for (std::size_t i = 0; i < half; ++i) {
        inv_freq[i] = std::pow(double(theta), -2.0 * double(i) / double(head_dim));
    }

    std::vector<float> data(positions * half * 2);
    for (std::size_t pos = 0; pos < positions; ++pos) {
        for (std::size_t i = 0; i < half; ++i) {
            const double angle = double(pos) * inv_freq[i];
            data[(pos * half + i) * 2] = static_cast<float>(std::cos(angle));
            data[(pos * half + i) * 2 + 1] = static_cast<float>(std::sin(angle));
        }
    }
    return Tensor::from_data<float>(context, data, {max_positions, head_dim / 2, std::int64_t{2}});
}

Tensor rope(const Tensor& x, const Tensor& table, std::int64_t start_pos, RopeStyle style) {
    const Shape& xs = x.shape();
    if (xs.size() < 3 || x.dtype() != DType::F32) {
        throw Error("rope: x must be f32 [..., seq, heads, head_dim], got " +
                    std::string(to_string(x.dtype())) + " " + to_string(xs));
    }
    const std::int64_t dim = xs[xs.size() - 1];
    const std::int64_t heads = xs[xs.size() - 2];
    const std::int64_t seq = xs[xs.size() - 3];
    const Shape& ts = table.shape();
    if (table.dtype() != DType::F32 || ts.size() != 3 || ts[1] * 2 != dim || ts[2] != 2) {
        throw Error("rope: x " + to_string(xs) + " does not match the table " + to_string(ts) +
                    " from rope_table");
    }
    if (start_pos < 0 || start_pos + seq > ts[0]) {
        throw Error("rope: positions " + std::to_string(start_pos) + " to " +
                    std::to_string(start_pos + seq - 1) + " are outside the table's " +
                    std::to_string(ts[0]) + " positions");
    }

    detail::Runtime& runtime = TensorAccess::runtime(x);
    Tensor out = TensorAccess::empty(runtime, xs, DType::F32);
    if (out.numel() == 0) return out;

    const RopeParams params{static_cast<std::uint32_t>(out.numel() / 2),
                            static_cast<std::uint32_t>(dim / 2), static_cast<std::uint32_t>(heads),
                            static_cast<std::uint32_t>(seq), static_cast<std::uint32_t>(start_pos)};
    const std::array<const hal::Buffer*, 3> buffers{
        &TensorAccess::buffer(x), &TensorAccess::buffer(table), &TensorAccess::buffer(out)};
    const hal::ComputePipeline& pipeline =
        runtime.pipeline(shaders::rope, 3, sizeof(params),
                         {runtime.workgroup_width(), static_cast<std::uint32_t>(style)});
    runtime.stream.dispatch(pipeline, buffers, std::as_bytes(std::span{&params, 1}),
                            {runtime.workgroup_count(params.pairs), 1, 1});
    return out;
}

}  // namespace vkml
