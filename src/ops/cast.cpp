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
    if (dtype != DType::F32 || !source) {
        throw Error("cast: no conversion from " + std::string(to_string(x.dtype())) + " to " +
                    std::string(to_string(dtype)) + " yet");
    }

    detail::Runtime& runtime = TensorAccess::runtime(x);
    Tensor out = TensorAccess::empty(runtime, x.shape(), DType::F32);
    if (out.numel() == 0) return out;

    const CastParams params{static_cast<std::uint32_t>(out.numel())};
    const std::array<const hal::Buffer*, 2> buffers{&TensorAccess::buffer(x),
                                                    &TensorAccess::buffer(out)};
    const hal::ComputePipeline& pipeline = runtime.pipeline(
        "cast", shaders::cast, 2, sizeof(params), {runtime.workgroup_width(), *source});
    runtime.stream.dispatch(pipeline, buffers, std::as_bytes(std::span{&params, 1}),
                            {runtime.workgroup_count(params.count), 1, 1});
    return out;
}

}  // namespace vkml
