#pragma once

#include "vkml/tensor.hpp"

namespace vkml {

// Elementwise arithmetic on two tensors of the same shape and dtype, returning
// a new tensor. f32 only for now.
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);

// Matrix product of a [..., M, K] and b [K, N], giving [..., M, N]. Leading
// dimensions of a are extra rows, which is the shape of a linear layer. f32 only.
Tensor matmul(const Tensor& a, const Tensor& b);

}  // namespace vkml
