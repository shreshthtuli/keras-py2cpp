#!/bin/python3
import json
from keras import backend as K
from keras.models import Sequential
from keras.layers import (
    LocallyConnected1D, Conv2D, Dense, Flatten, Activation,
    MaxPooling2D, Dropout, BatchNormalization
)
from tensorflow import ConfigProto, Session

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(Session(config=config))

model = Sequential([
    LocallyConnected1D(15, 3, input_shape=(10, 17)),
    LocallyConnected1D(8, 5),
    Flatten(),
    Dense(16),
    BatchNormalization(),
    Dense(1),
])

model.compile(loss='mse', optimizer='adam')

for layer in model.layers:
    print()
    print('Name: ', type(layer).__name__)
    print('Config: ', json.dumps(layer.get_config(), indent=2, sort_keys=True))
    # for w in layer.get_weights():
    #    print('Weights: ', w.shape)
    if isinstance(layer, BatchNormalization):
        print('Beta: ', layer.beta)
        print('Gamma: ', layer.gamma)
        print('Moving Mean: ', layer.moving_mean)
        print('Moving Variance: ', layer.moving_variance)

    if isinstance(layer, LocallyConnected1D):
        print('Kernel: ', layer.kernel)
        print('Bias: ', layer.bias)

print(model.summary())
