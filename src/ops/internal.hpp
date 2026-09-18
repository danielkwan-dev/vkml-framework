#pragma once

#include <cstdint>

#include "vkml/tensor.hpp"

// Op variants with options the public API does not expose, for composite ops.
namespace vkml::detail {

// matmul or matmul_transposed. With b_group g > 1, a [B * g, M, K] and b of
// rank 3 with B batches: batch z of a is multiplied by batch z / g of b.
Tensor matmul(const char* name, const Tensor& a, const Tensor& b, bool b_transposed,
              std::int64_t b_group = 1);

struct CausalMask {
    std::int64_t q_len;   // rows per head
    std::int64_t offset;  // position of query 0 among the columns (keys)
};

// softmax(scale * x) over the last dimension; with a mask, row r only sees
// columns <= (r % q_len) + offset.
Tensor softmax(const Tensor& x, float scale, const CausalMask* mask);

}  // namespace vkml::detail
