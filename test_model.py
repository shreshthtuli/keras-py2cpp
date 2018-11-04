import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

test_x = np.random.rand(100, 10).astype('f')
test_y = np.random.rand(100).astype('f')

input_shape = 10
num_classes = 1

model = Sequential()
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adamax')
model.fit(test_x, test_y, nb_epoch=1, verbose=False)

input = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
print(input.shape)
print (input)
print (model.predict(input))

from kerasify import export_model
export_model(model, 'yinsh.model')