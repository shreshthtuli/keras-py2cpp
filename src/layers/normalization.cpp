/*
 * Copyright (c) 2018 Shreshth Tuli
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/normalization.h"

namespace keras {
namespace layers {

BatchNormalization::BatchNormalization(Stream& file)
: weights_(file), biases_(file) {}

Tensor BatchNormalization::operator()(const Tensor& in) const noexcept {
    kassert(in.ndim());
    return in.fma(weights_, biases_);
}

} // namespace layers
} // namespace keras
