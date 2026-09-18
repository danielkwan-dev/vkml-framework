#include <cmath>
#include <cstdint>
#include <string>

#include "ops/internal.hpp"
#include "vkml/ops.hpp"

namespace vkml {

Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, bool causal) {
    const Shape& qs = q.shape();
    const Shape& ks = k.shape();
    if (qs.size() != 3 || ks.size() != 3 || v.shape().size() != 3) {
        throw Error("attention: q, k and v must be [heads, seq, head_dim], got " + to_string(qs) +
                    ", " + to_string(ks) + " and " + to_string(v.shape()));
    }
    const std::int64_t heads = qs[0];
    const std::int64_t q_len = qs[1];
    const std::int64_t head_dim = qs[2];
    const std::int64_t kv_heads = ks[0];
    const std::int64_t kv_len = ks[1];
    if (ks[2] != head_dim || v.shape() != ks) {
        throw Error("attention: k " + to_string(ks) + " and v " + to_string(v.shape()) +
                    " must match each other and q's head_dim in " + to_string(qs));
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
    const Tensor scores = detail::matmul("attention", q, k, true, group);  // [heads, q_len, kv_len]
    const detail::CausalMask mask{q_len, kv_len - q_len};
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const Tensor probs = detail::softmax(scores, scale, causal ? &mask : nullptr);
    return detail::matmul("attention", probs, v, false, group);
}

}  // namespace vkml
