#include <array>
#include <cstdint>
#include <string>

#include <vkml_shaders/cast.spv.hpp>

#include "core/runtime.hpp"
#include "core/storage.hpp"
#include "ops/internal.hpp"
#include "vkml/ops.hpp"

namespace vkml {

namespace {

using detail::TensorAccess;

struct CastParams {
    std::uint32_t count;
};

}  // namespace

Tensor cast(const Tensor& x, DType dtype) {
    const auto source = detail::shader_type(x.dtype());
    const auto target = detail::shader_type(dtype);
    // Widening to f32 from any float type, or narrowing f32 to a 16-bit one.
    if (!source || !target || (dtype != DType::F32 && x.dtype() != DType::F32)) {
        throw Error("cast: no conversion from " + std::string(to_string(x.dtype())) + " to " +
                    std::string(to_string(dtype)) + " yet");
    }

    detail::Runtime& runtime = TensorAccess::runtime(x);
    Tensor out = TensorAccess::empty(runtime, x.shape(), dtype);
    if (out.numel() == 0) return out;

    const CastParams params{static_cast<std::uint32_t>(out.numel())};
    const std::uint64_t invocations = dtype == DType::F32 ? params.count : (params.count + 1) / 2;
    const std::array<const hal::Buffer*, 2> buffers{&TensorAccess::buffer(x),
                                                    &TensorAccess::buffer(out)};
    const hal::ComputePipeline& pipeline = runtime.pipeline(
        "cast", shaders::cast, 2, sizeof(params), {runtime.workgroup_width(), *source, *target});
    runtime.stream.dispatch(pipeline, buffers, std::as_bytes(std::span{&params, 1}),
                            {runtime.workgroup_count(invocations), 1, 1});
    return out;
}

}  // namespace vkml
