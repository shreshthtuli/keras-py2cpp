﻿/*
 * Copyright (c) 2018 Shreshth Tuli
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layer.h"

namespace keras {
namespace layers {

class BatchNormalization final : public Layer<BatchNormalization> {
    Tensor weights_;
    Tensor biases_;

public:
    BatchNormalization(Stream& file);
    Tensor operator()(const Tensor& in) const noexcept override;
};

} // namespace layers
} // namespace keras
