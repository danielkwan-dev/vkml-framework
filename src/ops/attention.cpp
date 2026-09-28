#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>

#include <vkml_shaders/attend.spv.hpp>
#include <vkml_shaders/attend_merge.spv.hpp>
#include <vkml_shaders/write_rows.spv.hpp>

#include "core/runtime.hpp"
#include "core/storage.hpp"
#include "ops/internal.hpp"
#include "vkml/ops.hpp"

namespace vkml {

namespace {

using detail::TensorAccess;

struct AttendParams {
    std::uint32_t kv_len;
    std::uint32_t capacity;
    std::uint32_t group;
    std::uint32_t chunks;
    float scale;
};

struct MergeParams {
    std::uint32_t chunks;
};

constexpr std::int64_t kAttendChunk = 128;  // keys per workgroup in shaders/attend.comp
constexpr std::int64_t kAttendMaxHeadDim = 128;

// Attention for one query per head, fused and split over chunks of keys
// (shaders/attend.comp, then attend_merge.comp): q [heads, 1, head_dim]
// against the first kv_len rows of the caches. Every key is visible to a
// single query placed after them, so causal masking changes nothing.
Tensor attend_one_query(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
                        std::int64_t kv_len) {
    const std::int64_t heads = q.shape()[0], head_dim = q.shape()[2];
    const std::int64_t capacity = k_cache.shape()[1];
    const std::int64_t chunks = (kv_len + kAttendChunk - 1) / kAttendChunk;
    detail::Runtime& runtime = TensorAccess::runtime(q);
    const Tensor partial = TensorAccess::empty(runtime, {heads, chunks, head_dim}, DType::F32);
    const Tensor stats = TensorAccess::empty(runtime, {heads, chunks, 2}, DType::F32);
    Tensor out = TensorAccess::empty(runtime, {heads, 1, head_dim}, DType::F32);

    const auto u32 = [](std::int64_t x) { return static_cast<std::uint32_t>(x); };
    const AttendParams params{u32(kv_len), u32(capacity), u32(heads / k_cache.shape()[0]),
                              u32(chunks), 1.0f / std::sqrt(static_cast<float>(head_dim))};
    const std::array<const hal::Buffer*, 5> buffers{
        &TensorAccess::buffer(q), &TensorAccess::buffer(k_cache), &TensorAccess::buffer(v_cache),
        &TensorAccess::buffer(partial), &TensorAccess::buffer(stats)};
    runtime.stream.dispatch(
        runtime.pipeline("attend", shaders::attend, 5, sizeof(params), {u32(head_dim)}), buffers,
        std::as_bytes(std::span{&params, 1}), {u32(chunks), u32(heads), 1});

    const MergeParams merge{u32(chunks)};
    const std::array<const hal::Buffer*, 3> merge_buffers{
        &TensorAccess::buffer(partial), &TensorAccess::buffer(stats), &TensorAccess::buffer(out)};
    runtime.stream.dispatch(
        runtime.pipeline("attend_merge", shaders::attend_merge, 3, sizeof(merge), {u32(head_dim)}),
        merge_buffers, std::as_bytes(std::span{&merge, 1}), {u32(heads), 1, 1});
    return out;
}

struct CopyRowsParams {
    std::uint32_t count;
    std::uint32_t block;
    std::uint32_t src_stride;
    std::uint32_t src_start;
    std::uint32_t dst_stride;
    std::uint32_t dst_start;
};

// Copies rows src_row.. of every batch of src to rows dst_row.. of dst, rows
// of each: both [B, *, width] of 32-bit elements (shaders/write_rows.comp).
void copy_rows(const Tensor& src, std::int64_t src_row, const Tensor& dst, std::int64_t dst_row,
               std::int64_t rows) {
    const std::int64_t batches = src.shape()[0], width = src.shape()[2];
    const auto u32 = [](std::int64_t x) { return static_cast<std::uint32_t>(x); };
    const CopyRowsParams params{u32(batches * rows * width), u32(rows * width),
                                u32(src.shape()[1] * width), u32(src_row * width),
                                u32(dst.shape()[1] * width), u32(dst_row * width)};
    if (params.count == 0) return;
    detail::Runtime& runtime = TensorAccess::runtime(dst);
    const std::array<const hal::Buffer*, 2> buffers{&TensorAccess::buffer(src),
                                                    &TensorAccess::buffer(dst)};
    const hal::ComputePipeline& pipeline = runtime.pipeline(
        "write_rows", shaders::write_rows, 2, sizeof(params), {runtime.workgroup_width()});
    runtime.stream.dispatch(pipeline, buffers, std::as_bytes(std::span{&params, 1}),
                            {runtime.workgroup_count(params.count), 1, 1});
}

// Scores, softmax and weighted sum of v for q against the first kv_len rows
// of the caches, with query i at key position kv_len - q_len + i.
Tensor attention_block(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
                       std::int64_t kv_len, bool causal) {
    const std::int64_t q_len = q.shape()[1];
    const std::int64_t group = q.shape()[0] / k_cache.shape()[0];
    // [heads, q_len, kv_len]
    const Tensor scores = detail::matmul("attention", q, k_cache, true, group, kv_len);
    const detail::CausalMask mask{q_len, kv_len - q_len};
    const float scale = 1.0f / std::sqrt(static_cast<float>(q.shape()[2]));
    const Tensor probs = detail::softmax(scores, scale, causal ? &mask : nullptr);
    return detail::matmul("attention", probs, v_cache, false, group, kv_len);
}

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
    copy_rows(src, 0, dst, start, ss[1]);
}

Tensor detail::read_rows(const Tensor& src, std::int64_t start, std::int64_t count) {
    const Shape& ss = src.shape();
    if (ss.size() != 3 || start < 0 || count < 0 || start + count > ss[1] ||
        element_size(src.dtype()) != 4) {
        throw Error("read_rows: cannot read rows " + std::to_string(start) + ".." +
                    std::to_string(start + count) + " of " + std::string(to_string(src.dtype())) +
                    " " + to_string(ss));
    }
    Tensor out =
        TensorAccess::empty(TensorAccess::runtime(src), {ss[0], count, ss[2]}, src.dtype());
    copy_rows(src, start, out, 0, count);
    return out;
}

Tensor detail::attention(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
                         std::int64_t kv_len, bool causal, std::int64_t max_score_bytes) {
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

    if (q_len == 1 && kv_len > 0 && head_dim % 4 == 0 && head_dim <= kAttendMaxHeadDim &&
        heads <= std::int64_t(TensorAccess::runtime(q).device.info().max_workgroup_count[1])) {
        return attend_one_query(q, k_cache, v_cache, kv_len);
    }

    // A long prompt's scores [heads, q_len, kv_len] would take hundreds of MB
    // per layer (288 for 1500 tokens of TinyLlama), allocated afresh each
    // time. Chunks of queries keep them small and, causal, stop at the last
    // key the chunk's last query sees, skipping most of the masked half.
    const std::int64_t row_bytes = heads * std::max<std::int64_t>(kv_len, 1) * 4;
    std::int64_t chunk = std::max<std::int64_t>(1, max_score_bytes / row_bytes);
    if (chunk >= 64) chunk -= chunk % 64;  // whole blocks of the tiled matmul
    if (q_len <= chunk) return attention_block(q, k_cache, v_cache, kv_len, causal);

    Tensor out = TensorAccess::empty(TensorAccess::runtime(q), qs, DType::F32);
    const std::int64_t first_position = kv_len - q_len;  // of query 0 among the keys
    for (std::int64_t q0 = 0; q0 < q_len; q0 += chunk) {
        const std::int64_t rows = std::min(chunk, q_len - q0);
        const std::int64_t keys = causal ? first_position + q0 + rows : kv_len;
        copy_rows(attention_block(read_rows(q, q0, rows), k_cache, v_cache, keys, causal), 0, out,
                  q0, rows);
    }
    return out;
}

Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, bool causal) {
    const std::int64_t kv_len = k.shape().size() == 3 ? k.shape()[1] : 0;
    return detail::attention(q, k, v, kv_len, causal);
}

}  // namespace vkml
