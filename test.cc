/*
 * Copyright (c) 2018 Shreshth Tuli
 *
 * MIT License, see LICENSE file.
 */

#include "keras_model.h"

int main() {
    // Initialize model.
    KerasModel model;
    KASSERT(model.LoadModel("yinsh.model"), "Failed to load model");

    // Create a 1D Tensor on length 10 for input data.
    Tensor in(10);
    in.data_ = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Run prediction.
    Tensor out(1);
    KASSERT(model.Apply(&in, &out), "Failed to apply");
    out.Print();
    return 0;
}