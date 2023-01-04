import argparse
import importlib
import numpy as np
import os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

#from medium import build_model

archs = [s.split('.')[0] for s in os.listdir('archs') if s[0:1] != '_']
print(archs)

parser = argparse.ArgumentParser()
parser.add_argument('arch', type=str, choices=archs, help='Type of neural network architectures')
args = parser.parse_args()

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

#model = build_model(input_shape)
model = importlib.import_module('archs.' + args.arch).build_model(input_shape)
model.summary()

# Compile model
model.compile(loss='mae', optimizer='adam')

# Train model
model.fit(X, Y, batch_size=batch_size, epochs=epochs)

model.save('mymodel')
