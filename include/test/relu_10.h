/* Autogenerated file, DO NOT EDIT */
#pragma once

#include "keras/model.h"

namespace test {

inline auto relu_10() {
    printf("TEST relu_10\n");

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
    keras::Tensor in{10};
    in.data_ = {0.89932585, 0.97299117, 0.32186216, 0.7445666, 0.26783466, 0.982939, 0.8939688, 0.63179505, 0.37665573, 0.9020204};

    keras::Tensor target{10};
    target.data_ = {0.2845814, 0.09887414, 1.4839032, 0., 0., 0.30709586, 0., 0.4679813, 0.69397587, 0.67023253};
#pragma GCC diagnostic pop

    auto [model, load_time] = keras::timeit(keras::Model::load, "C:\Users\user\Desktop\keras-py2cpp\models\relu_10.model");
    auto [output, apply_time] = keras::timeit(model, in);

    for (size_t i = 0; i < target.dims_[0]; ++i)
        kassert_eq(target(i), output(i), 1e-6);

    return std::make_tuple(load_time, apply_time);
}

} // namespace test
