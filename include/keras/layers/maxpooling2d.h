/*
 * Copyright (c) 2018 Shreshth Tuli
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layer.h"

namespace keras {
namespace layers {

class MaxPooling2D final : public Layer<MaxPooling2D> {
    unsigned pool_size_y_{0};
    unsigned pool_size_x_{0};

public:
    MaxPooling2D(Stream& file);
    Tensor operator()(const Tensor& in) const noexcept override;
};

} // namespace layers
} // namespace keras
