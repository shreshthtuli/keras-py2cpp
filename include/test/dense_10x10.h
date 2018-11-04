/* Autogenerated file, DO NOT EDIT */
#pragma once

#include "keras/model.h"

namespace test {

inline auto dense_10x10() {
    printf("TEST dense_10x10\n");

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
    keras::Tensor in{10};
    in.data_ = {0.9194567, 0.34545597, 0.5294414, 0.40176404, 0.5805263, 0.28934142, 0.96964633, 0.6926722, 0.612759, 0.12978625};

    keras::Tensor target{1};
    target.data_ = {-0.44967902};
#pragma GCC diagnostic pop

    auto [model, load_time] = keras::timeit(keras::Model::load, "C:\Users\user\Desktop\keras-py2cpp\models\dense_10x10.model");
    auto [output, apply_time] = keras::timeit(model, in);

    for (size_t i = 0; i < target.dims_[0]; ++i)
        kassert_eq(target(i), output(i), 1e-6);

    return std::make_tuple(load_time, apply_time);
}

} // namespace test
