/* Autogenerated file, DO NOT EDIT */
#pragma once

#include "keras/model.h"

namespace test {

inline auto conv_hard_sigmoid_2x2() {
    printf("TEST conv_hard_sigmoid_2x2\n");

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
    keras::Tensor in{2, 2, 1};
    in.data_ = {0.19852342, 0.6710748, 0.41525805, 0.78936124};

    keras::Tensor target{1};
    target.data_ = {-0.20380606};
#pragma GCC diagnostic pop

    auto [model, load_time] = keras::timeit(keras::Model::load, "C:\Users\user\Desktop\keras-py2cpp\models\conv_hard_sigmoid_2x2.model");
    auto [output, apply_time] = keras::timeit(model, in);

    for (size_t i = 0; i < target.dims_[0]; ++i)
        kassert_eq(target(i), output(i), 1e-6);

    return std::make_tuple(load_time, apply_time);
}

} // namespace test
