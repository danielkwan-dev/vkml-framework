#include <array>
#include <cmath>
#include <cstdint>
#include <string>

#include <vkml_shaders/write_rows.spv.hpp>

#include "core/runtime.hpp"
#include "core/storage.hpp"
#include "ops/internal.hpp"
#include "vkml/ops.hpp"

namespace vkml {

namespace {

using detail::TensorAccess;

struct WriteRowsParams {
    std::uint32_t count;
    std::uint32_t block;
    std::uint32_t dst_stride;
    std::uint32_t dst_start;
};

}  // namespace

void detail::write_rows(const Tensor& dst, const Tensor& src, std::int64_t start) {
    const Shape& ds = dst.shape();
    const Shape& ss = src.shape();
    if (ds.size() != 3 || ss.size() != 3 || ds[0] != ss[0] || ds[2] != ss[2] || start < 0 ||
        start + ss[1] > ds[1] || element_size(dst.dtype()) != 4 || dst.dtype() != src.dtype()) {
        throw Error("write_rows: cannot write " + std::string(to_string(src.dtype())) + " " +
                    to_string(ss) + " at row " + std::to_string(start) + " of " +
                    std::string(to_string(dst.dtype())) + " " + to_string(ds));
    }
    if (src.numel() == 0) return;

    detail::Runtime& runtime = TensorAccess::runtime(dst);
    const WriteRowsParams params{
        static_cast<std::uint32_t>(src.numel()), static_cast<std::uint32_t>(ss[1] * ss[2]),
        static_cast<std::uint32_t>(ds[1] * ds[2]), static_cast<std::uint32_t>(start * ds[2])};
    const std::array<const hal::Buffer*, 2> buffers{&TensorAccess::buffer(src),
                                                    &TensorAccess::buffer(dst)};
    const hal::ComputePipeline& pipeline = runtime.pipeline(
        "write_rows", shaders::write_rows, 2, sizeof(params), {runtime.workgroup_width()});
    runtime.stream.dispatch(pipeline, buffers, std::as_bytes(std::span{&params, 1}),
                            {runtime.workgroup_count(params.count), 1, 1});
}

Tensor detail::attention(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
                         std::int64_t kv_len, bool causal) {
    const Shape& qs = q.shape();
    const Shape& ks = k_cache.shape();
    if (qs.size() != 3 || ks.size() != 3 || v_cache.shape().size() != 3) {
        throw Error("attention: q, k and v must be [heads, seq, head_dim], got " + to_string(qs) +
                    ", " + to_string(ks) + " and " + to_string(v_cache.shape()));
    }
    const std::int64_t heads = qs[0];
    const std::int64_t q_len = qs[1];
    const std::int64_t head_dim = qs[2];
    const std::int64_t kv_heads = ks[0];
    if (ks[2] != head_dim || v_cache.shape() != ks) {
        throw Error("attention: k " + to_string(ks) + " and v " + to_string(v_cache.shape()) +
                    " must match each other and q's head_dim in " + to_string(qs));
    }
    if (kv_len < 0 || kv_len > ks[1]) {
        throw Error("attention: " + std::to_string(kv_len) + " keys do not fit the cache " +
                    to_string(ks));
    }
    if (kv_heads == 0 || heads % kv_heads != 0) {
        throw Error("attention: q's " + std::to_string(heads) + " heads must be a multiple of " +
                    "k and v's " + std::to_string(kv_heads));
    }
    if (causal && q_len > kv_len) {
        throw Error("attention: " + std::to_string(q_len) + " queries is more queries than the " +
                    std::to_string(kv_len) + " keys a causal mask can place them among");
    }

    const std::int64_t group = heads / kv_heads;
    // [heads, q_len, kv_len]
    const Tensor scores = detail::matmul("attention", q, k_cache, true, group, kv_len);
    const detail::CausalMask mask{q_len, kv_len - q_len};
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const Tensor probs = detail::softmax(scores, scale, causal ? &mask : nullptr);
    return detail::matmul("attention", probs, v_cache, false, group, kv_len);
}

Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, bool causal) {
    const std::int64_t kv_len = k.shape().size() == 3 ? k.shape()[1] : 0;
    return detail::attention(q, k, v, kv_len, causal);
}

}  // namespace vkml
