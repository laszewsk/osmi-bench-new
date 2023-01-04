import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, Conv2DTranspose, UpSampling2D, Cropping2D
from tensorflow.keras.layers import Reshape, Lambda, MaxPooling1D, Conv1D, Conv2DTranspose, BatchNormalization, Add, ZeroPadding2D, Conv2D, concatenate



def build_model(input_shape, filters=(3, 3), actfn='elu', padding='same'):

	input_layer1 = Input(shape=input_shape, name='input')
	pool = (2,2)

	conv2 = Conv2D(32, kernel_size=filters, padding=padding, activation=actfn)(input_layer1)

	conv3 = Conv2D(64, kernel_size=filters, padding=padding, activation=actfn)(conv2)
	max3 = MaxPooling2D(pool_size=pool)(conv3)

	conv4 = Conv2D(128, kernel_size=filters, padding=padding, activation=actfn)(max3)
	max4 = MaxPooling2D(pool_size=pool)(conv4)

	conv5 = Conv2D(256, kernel_size=filters, padding=padding, activation=actfn)(max4)
	max5 = MaxPooling2D(pool_size=(2, 2))(conv5)
	crop5 = Cropping2D(cropping=((2, 2), (1, 1)))(max5)

	conv6 = Conv2D(512, kernel_size=filters, padding=padding, activation=actfn)(crop5)
	max6 = MaxPooling2D(pool_size=pool)(conv6)

	conv7 = Conv2D(1024, kernel_size=filters, padding=padding, activation=actfn)(max6)
	max7 = MaxPooling2D(pool_size=(4,4))(conv7)

	flatencoder1 = Flatten()(max7)
	flat_units = int(max7.shape[3])

	# Dense fully connected regression
	dense1 = Dense(flat_units, activation=actfn)(flatencoder1)

	# Reshape before decoding
	dmap1 = Reshape((1, 1, flat_units))(dense1)

	cpdconv1 = Conv2DTranspose(flat_units, kernel_size=filters, strides=(1, 1),\
                               padding=padding, activation=actfn)(dmap1)

	cpdconv2 = Conv2DTranspose(512, kernel_size=filters, strides=(2, 2),\
                               padding=padding, activation=actfn)(cpdconv1)

	cpdconv3 = Conv2DTranspose(256, kernel_size=filters, strides=(2, 2),\
                               padding=padding, activation=actfn)(cpdconv2)

	cpdconv4 = Conv2DTranspose(128, kernel_size=filters, strides=(2, 2),\
                               padding=padding, activation=actfn)(cpdconv3)

	cpdconv5 = Conv2DTranspose(64, kernel_size=filters, strides=(4, 4),\
                               padding=padding, activation=actfn)(cpdconv4)
	cpcrop5 = Cropping2D(cropping=((4,3),(6,6)))(cpdconv5)

	cpdconv6 = Conv2DTranspose(32, kernel_size=filters, strides=(2, 2),\
                               padding=padding, activation=actfn)(cpcrop5)
	cppad6 = ZeroPadding2D(padding=((0,0),(1,0)))(cpdconv6)

	cpdconv7 = Conv2DTranspose(16, kernel_size=filters, strides=(2, 2),\
                               padding=padding, activation=actfn)(cppad6)
	cppad7 = ZeroPadding2D(padding=((1,0),(0,0)))(cpdconv7)

	cpdconv8 = Conv2DTranspose(8, kernel_size=filters, strides=(1, 1),\
                               padding=padding, activation=actfn)(cppad7)

	cpdconv9 = Conv2DTranspose(4, kernel_size=filters,
                               padding=padding, activation=actfn)(cpdconv8)

	cpdconv10 = Conv2DTranspose(2, kernel_size=filters,
                                padding=padding, activation=actfn)(cpdconv9)

	cpdconv11 = Conv2DTranspose(1, kernel_size=filters,
                                padding=padding, activation='linear')(cpdconv10)

	return Model(inputs=[input_layer1], outputs=[cpdconv11])
