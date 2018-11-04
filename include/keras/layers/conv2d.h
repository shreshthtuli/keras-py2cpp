/*
 * Copyright (c) 2018 Shreshth Tuli
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layers/activation.h"

namespace keras {
namespace layers {

class Conv2D final : public Layer<Conv2D> {
    Tensor weights_;
    Tensor biases_;
    Activation activation_;

public:
    Conv2D(Stream& file);
    Tensor operator()(const Tensor& in) const noexcept override;
};

} // namespace layers
} // namespace keras
