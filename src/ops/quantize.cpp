#include <array>
#include <cstdint>
#include <string>

#include <vkml_shaders/dequantize.spv.hpp>
#include <vkml_shaders/quantize.spv.hpp>

#include "core/runtime.hpp"
#include "core/storage.hpp"
#include "ops/internal.hpp"
#include "vkml/ops.hpp"

namespace vkml {

namespace {

using detail::TensorAccess;

constexpr std::int64_t kBlock = 32;  // Q8_BLOCK and Q4_BLOCK in shaders/include/halves.glsl

struct CountParams {
    std::uint32_t count;
};

QuantizedMatrix quantize(const char* name, const Tensor& w, QuantType quant) {
    const Shape& s = w.shape();
    const auto type = detail::shader_type(w.dtype());
    if (s.size() != 2 || !type) {
        throw Error(std::string(name) + ": expected a float matrix, got " +
                    std::string(to_string(w.dtype())) + " " + to_string(s));
    }
    if (s[1] % kBlock != 0) {
        throw Error(std::string(name) + ": columns must be a multiple of 32, got " + to_string(s));
    }
    detail::Runtime& runtime = TensorAccess::runtime(w);
    const std::int64_t per_word = detail::quant_values_per_word(quant);
    QuantizedMatrix q{.values = TensorAccess::empty(runtime, {s[0], s[1] / per_word}, DType::I32),
                      .scales = TensorAccess::empty(runtime, {s[0], s[1] / kBlock}, DType::F32),
                      .rows = s[0],
                      .cols = s[1],
                      .type = quant};
    if (w.numel() == 0) return q;

    const CountParams params{static_cast<std::uint32_t>(w.numel() / kBlock)};
    const std::array<const hal::Buffer*, 3> buffers{
        &TensorAccess::buffer(w), &TensorAccess::buffer(q.values), &TensorAccess::buffer(q.scales)};
    const hal::ComputePipeline& pipeline =
        runtime.pipeline("quantize", shaders::quantize, 3, sizeof(params),
                         {runtime.workgroup_width(), *type, detail::quant_shader_type(quant)});
    runtime.stream.dispatch(pipeline, buffers, std::as_bytes(std::span{&params, 1}),
                            {runtime.workgroup_count(params.count), 1, 1});
    return q;
}

}  // namespace

QuantizedMatrix quantize_q8(const Tensor& w) { return quantize("quantize_q8", w, QuantType::q8_0); }

QuantizedMatrix quantize_q4(const Tensor& w) { return quantize("quantize_q4", w, QuantType::q4_0); }

Tensor dequantize(const QuantizedMatrix& q) {
    detail::Runtime& runtime = TensorAccess::runtime(q.values);
    Tensor out = TensorAccess::empty(runtime, {q.rows, q.cols}, DType::F32);
    if (out.numel() == 0) return out;
    const CountParams params{static_cast<std::uint32_t>(out.numel())};
    const std::array<const hal::Buffer*, 3> buffers{&TensorAccess::buffer(q.values),
                                                    &TensorAccess::buffer(q.scales),
                                                    &TensorAccess::buffer(out)};
    const hal::ComputePipeline& pipeline =
        runtime.pipeline("dequantize", shaders::dequantize, 3, sizeof(params),
                         {runtime.workgroup_width(), detail::quant_shader_type(q.type)});
    runtime.stream.dispatch(pipeline, buffers, std::as_bytes(std::span{&params, 1}),
                            {runtime.workgroup_count(params.count), 1, 1});
    return out;
}

}  // namespace vkml
