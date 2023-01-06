"""usage: python train.py medium"""
import argparse
import importlib
import numpy as np
import os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

archs = [s.split('.')[0] for s in os.listdir('archs') if s[0:1] != '_']
print(archs)

parser = argparse.ArgumentParser()
parser.add_argument('arch', type=str, choices=archs, help='Type of neural network architectures')
args = parser.parse_args()

# parameters
samples = 100 
epochs = 5
batch_size = 32

# compute synthetic data for X and Y
# small model
input_shape = (8, 48)
output_shape = (2, 12)
# medium model
#input_shape = (101, 82, 9)
#output_shape = (101, 82)
# large model
#input_shape = (2, 101, 82, 9)
#output_shape = (2, 101, 82, 1)
X = np.random.rand(samples, *input_shape)
Y = np.random.rand(samples, *output_shape)

# define model
model = importlib.import_module('archs.' + args.arch).build_model(input_shape)
model.summary()

# compile model
model.compile(loss='mae', optimizer='adam')

# train model
model.fit(X, Y, batch_size=batch_size, epochs=epochs)

model.save('mymodel')
