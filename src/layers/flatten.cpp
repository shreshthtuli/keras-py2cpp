/*
 * Copyright (c) 2018 Shreshth Tuli
 *
 * MIT License, see LICENSE file.
 */
#include "keras/layers/flatten.h"

namespace keras {
namespace layers {

Tensor Flatten::operator()(const Tensor& in) const noexcept {
    return Tensor(in).flatten();
}

} // namespace layers
} // namespace keras
