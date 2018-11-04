/*
 * Copyright (c) 2018 Shreshth Tuli
 *
 * MIT License, see LICENSE file.
 */
#pragma once

#include "keras/layer.h"

namespace keras {
namespace layers {

class Flatten final : public Layer<Flatten> {
public:
    using Layer<Flatten>::Layer;
    Tensor operator()(const Tensor& in) const noexcept override;
};

} // namespace layers
} // namespace keras
