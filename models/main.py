import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from medium import build_model

# parameters
samples = 100 
epochs = 5
batch_size = 32

# compute synthetic data for X and y (0's and 1's)
input_shape = (101, 82, 9)
output_shape = (101, 82)
X = np.random.rand(samples, *input_shape)
print(X.shape)
Y = np.random.rand(samples, *output_shape)

# Define architecture
units = 512
act_func = 'selu'
#layer_func = layers.Dense
#model = keras.Sequential()
#model.add(layer_func(units, activation=act_func, input_shape=(num_feats,)))
#model.add(layer_func(units, activation=act_func))
#model.add(layer_func(units, activation=act_func))
#model.add(layers.Dense(1, activation='sigmoid'))

model = build_model(input_shape)
model.summary()

# Compile model
model.compile(loss='mae', optimizer='adam')

# Train model
model.fit(X, Y, batch_size=batch_size, epochs=epochs)

model.save('mymodel')
