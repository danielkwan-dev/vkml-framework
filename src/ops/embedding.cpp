#include <array>
#include <cstdint>
#include <string>

#include <vkml_shaders/embedding.spv.hpp>
#include <vkml_shaders/embedding_quantized.spv.hpp>

#include "core/runtime.hpp"
#include "core/storage.hpp"
#include "ops/internal.hpp"
#include "vkml/ops.hpp"

namespace vkml {

namespace {

using detail::TensorAccess;

struct EmbeddingParams {
    std::uint32_t count;
    std::uint32_t dim;
    std::uint32_t vocab;
};

}  // namespace

Tensor embedding(const Tensor& table, const Tensor& ids) {
    const auto table_type = detail::shader_type(table.dtype());
    if (table.shape().size() != 2 || !table_type) {
        throw Error("embedding: table must be a float [vocab, dim] matrix, got " +
                    std::string(to_string(table.dtype())) + " " + to_string(table.shape()));
    }
    if (ids.dtype() != DType::I32) {
        throw Error("embedding: ids must be i32, got " + std::string(to_string(ids.dtype())));
    }

    const std::int64_t vocab = table.shape()[0];
    const std::int64_t dim = table.shape()[1];
    Shape out_shape = ids.shape();
    out_shape.push_back(dim);

    detail::Runtime& runtime = TensorAccess::runtime(ids);
    Tensor out = TensorAccess::empty(runtime, std::move(out_shape), DType::F32);
    if (out.numel() == 0) return out;
    if (vocab == 0) {  // every id is out of range, and the table has no buffer to bind
        runtime.fill_zeros(TensorAccess::buffer(out));
        return out;
    }

    const EmbeddingParams params{static_cast<std::uint32_t>(out.numel()),
                                 static_cast<std::uint32_t>(dim),
                                 static_cast<std::uint32_t>(vocab)};
    const std::array<const hal::Buffer*, 3> buffers{
        &TensorAccess::buffer(table), &TensorAccess::buffer(ids), &TensorAccess::buffer(out)};
    const hal::ComputePipeline& pipeline =
        runtime.pipeline("embedding", shaders::embedding, 3, sizeof(params),
                         {runtime.workgroup_width(), *table_type});
    runtime.stream.dispatch(pipeline, buffers, std::as_bytes(std::span{&params, 1}),
                            {runtime.workgroup_count(params.count), 1, 1});
    return out;
}

Tensor embedding(const QuantizedMatrix& table, const Tensor& ids) {
    if (ids.dtype() != DType::I32) {
        throw Error("embedding: ids must be i32, got " + std::string(to_string(ids.dtype())));
    }
    Shape out_shape = ids.shape();
    out_shape.push_back(table.cols);

    detail::Runtime& runtime = TensorAccess::runtime(ids);
    Tensor out = TensorAccess::empty(runtime, std::move(out_shape), DType::F32);
    if (out.numel() == 0) return out;
    if (table.rows == 0) {
        runtime.fill_zeros(TensorAccess::buffer(out));
        return out;
    }

    const EmbeddingParams params{static_cast<std::uint32_t>(out.numel()),
                                 static_cast<std::uint32_t>(table.cols),
                                 static_cast<std::uint32_t>(table.rows)};
    const std::array<const hal::Buffer*, 4> buffers{
        &TensorAccess::buffer(table.values), &TensorAccess::buffer(table.scales),
        &TensorAccess::buffer(ids), &TensorAccess::buffer(out)};
    const hal::ComputePipeline& pipeline =
        runtime.pipeline("embedding_quantized", shaders::embedding_quantized, 4, sizeof(params),
                         {runtime.workgroup_width(), detail::quant_shader_type(table.type)});
    runtime.stream.dispatch(pipeline, buffers, std::as_bytes(std::span{&params, 1}),
                            {runtime.workgroup_count(params.count), 1, 1});
    return out;
}

}  // namespace vkml
