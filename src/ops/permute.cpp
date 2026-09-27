#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <vkml_shaders/permute.spv.hpp>

#include "core/runtime.hpp"
#include "core/storage.hpp"
#include "vkml/ops.hpp"

namespace vkml {

namespace {

using detail::TensorAccess;

constexpr std::size_t kMaxRank = 6;  // matches shaders/permute.comp

struct PermuteParams {
    std::uint32_t count;
    std::array<std::uint32_t, kMaxRank> out_shape;
    std::array<std::uint32_t, kMaxRank> src_strides;
};

std::string dims_string(const std::vector<int>& dims) {
    return to_string(Shape(dims.begin(), dims.end()));
}

}  // namespace

Tensor permute(const Tensor& x, std::vector<int> dims) {
    const Shape& shape = x.shape();
    const std::size_t rank = shape.size();
    if (rank > kMaxRank) {
        throw Error("permute: at most " + std::to_string(kMaxRank) + " dimensions, got " +
                    to_string(shape));
    }
    std::vector<int> sorted = dims;
    std::ranges::sort(sorted);
    std::vector<int> identity(rank);
    for (std::size_t i = 0; i < rank; ++i) identity[i] = static_cast<int>(i);
    if (sorted != identity) {
        throw Error("permute: " + dims_string(dims) + " is not a permutation of the " +
                    std::to_string(rank) + " dimensions of " + to_string(shape));
    }
    if (element_size(x.dtype()) != 4) {
        throw Error("permute: no kernel for " + std::string(to_string(x.dtype())) + " yet");
    }

    std::vector<std::uint64_t> in_strides(rank, 1);
    for (std::size_t i = rank; i-- > 1;) {
        in_strides[i - 1] = in_strides[i] * static_cast<std::uint64_t>(shape[i]);
    }
    Shape out_shape(rank);
    for (std::size_t i = 0; i < rank; ++i) out_shape[i] = shape[static_cast<std::size_t>(dims[i])];

    detail::Runtime& runtime = TensorAccess::runtime(x);
    Tensor out = TensorAccess::empty(runtime, out_shape, x.dtype());
    if (out.numel() == 0) return out;

    PermuteParams params{};
    params.count = static_cast<std::uint32_t>(out.numel());
    params.out_shape.fill(1);  // padding dimensions: size 1, stride 0
    const std::size_t pad = kMaxRank - rank;
    for (std::size_t i = 0; i < rank; ++i) {
        params.out_shape[pad + i] = static_cast<std::uint32_t>(out_shape[i]);
        params.src_strides[pad + i] =
            static_cast<std::uint32_t>(in_strides[static_cast<std::size_t>(dims[i])]);
    }

    const std::array<const hal::Buffer*, 2> buffers{&TensorAccess::buffer(x),
                                                    &TensorAccess::buffer(out)};
    const hal::ComputePipeline& pipeline = runtime.pipeline(
        "permute", shaders::permute, 2, sizeof(params), {runtime.workgroup_width()});
    runtime.stream.dispatch(pipeline, buffers, std::as_bytes(std::span{&params, 1}),
                            {runtime.workgroup_count(params.count), 1, 1});
    return out;
}

Tensor transpose(const Tensor& x, int dim0, int dim1) {
    const int rank = static_cast<int>(x.shape().size());
    for (int* d : {&dim0, &dim1}) {
        if (*d < -rank || *d >= rank) {
            throw Error("transpose: dimension " + std::to_string(*d) + " is out of range for " +
                        to_string(x.shape()));
        }
        if (*d < 0) *d += rank;
    }
    std::vector<int> dims(static_cast<std::size_t>(rank));
    for (int i = 0; i < rank; ++i) dims[static_cast<std::size_t>(i)] = i;
    std::swap(dims[static_cast<std::size_t>(dim0)], dims[static_cast<std::size_t>(dim1)]);
    return permute(x, std::move(dims));
}

}  // namespace vkml
