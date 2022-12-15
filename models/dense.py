import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# number of samples
num_samples = 100
num_feats = 1426

# compute synthetic data for X and y (0's and 1's)
X = np.random.rand(num_samples, num_feats)
n = num_samples//2
y = np.random.rand(num_samples)

# define architecture
units = 512
act_func = 'selu'
layer_func = layers.Dense
model = keras.Sequential()
model.add(layer_func(units, activation=act_func, input_shape=(num_feats,)))
model.add(layer_func(units, activation=act_func))
model.add(layer_func(units, activation=act_func))
model.add(layers.Dense(1, activation='sigmoid'))

# compile model
model.compile(loss='mae', optimizer='adam')

# train model
model.fit(X, y, batch_size=32, epochs=100)

model.save('mymodel/1')
