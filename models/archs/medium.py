import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, Conv2DTranspose, UpSampling2D, Cropping2D
from tensorflow.keras.layers import Reshape, Lambda, MaxPooling1D, Conv1D, Conv2DTranspose, BatchNormalization, Add, ZeroPadding2D, Conv2D, concatenate



def build_model(input_shape, filters=(3, 3), actfn='elu'):
	input_layer1 = Input(shape=input_shape, name='input')
	pool = (2,2)

	#(101,82,11)
	conv2 = Conv2D(32, kernel_size=filters,\
                   padding='same', activation='elu')(input_layer1)

	#(101,82,16)
	conv3 = Conv2D(64, kernel_size=filters,\
                   padding='same', activation='elu')(conv2)
	max3 = MaxPooling2D(pool_size=pool)(conv3)

	#(50,41,32)
	conv4 = Conv2D(128, kernel_size=filters,\
	padding='same', activation='elu')(max3)
	max4 = MaxPooling2D(pool_size=pool)(conv4)

	#(25,20,64)
	conv5 = Conv2D(256, kernel_size=filters,\
	padding='same', activation='elu')(max4)
	max5 = MaxPooling2D(pool_size=(2, 2))(conv5)
	crop5 = Cropping2D(cropping=((2,2),(1,1)))(max5)

	#(8,8,128)
	conv6 = Conv2D(512, kernel_size=filters,\
	padding='same', activation='elu')(crop5)
	max6 = MaxPooling2D(pool_size=pool)(conv6)

	#(4,4,256)
	conv7 = Conv2D(1024, kernel_size=filters,\
	padding='same', activation='elu')(max6)
	max7 = MaxPooling2D(pool_size=(4,4))(conv7)

	flatencoder1 = Flatten()(max7)
	flat_units = int(max7.shape[3])

	# Dense fully connected regression.
	dense1 = Dense(flat_units, activation = 'elu')(flatencoder1)
	#dense2 = Dense(flat_units, activation = 'elu')(flatencoder1)
	# Reshaping before decoding
	dmap1 = Reshape((1, 1, flat_units))(dense1)
	#dmap2 = Reshape((1, 1, flat_units))(dense2)
	#dmap3 = Reshape((1, 1, flat_units))(dense2)
	#dmap4 = Reshape((1, 1, flat_units))(dense2)
	#(1,1,1024)

	cpdconv1 = Conv2DTranspose(flat_units, kernel_size=filters, strides=(1, 1),\
	padding='same', activation='elu')(dmap1)

	#(1,1,1024)
	cpdconv2 = Conv2DTranspose(512, kernel_size=filters, strides=(2, 2),\
	padding='same', activation='elu')(cpdconv1)

	#(2,2,512)
	cpdconv3 = Conv2DTranspose(256, kernel_size=filters, strides=(2, 2),\
	padding='same', activation='elu')(cpdconv2)

	#(4,4,256)
	cpdconv4 = Conv2DTranspose(128, kernel_size=filters, strides=(2, 2),\
	padding='same', activation='elu')(cpdconv3)

	#(8,8,128)
	cpdconv5 = Conv2DTranspose(64, kernel_size=filters, strides=(4, 4),\
	padding='same', activation='elu')(cpdconv4)
	cpcrop5 = Cropping2D(cropping=((4,3),(6,6)))(cpdconv5)

	#(25,20,64)
	cpdconv6 = Conv2DTranspose(32, kernel_size=filters, strides=(2, 2),\
	padding='same', activation='elu')(cpcrop5)
	cppad6 = ZeroPadding2D(padding=((0,0),(1,0)))(cpdconv6)

	#(50,41,32)
	cpdconv7 = Conv2DTranspose(16, kernel_size=filters, strides=(2, 2),\
	padding='same', activation='elu')(cppad6)
	cppad7 = ZeroPadding2D(padding=((1,0),(0,0)))(cpdconv7)

	#(101,82,16)
	cpdconv8 = Conv2DTranspose(8, kernel_size=filters, strides=(1, 1),\
	padding='same', activation='elu')(cppad7)

	#(101,82,8)
	cpdconv9 = Conv2DTranspose(4, kernel_size=filters,
	padding='same', activation='elu')(cpdconv8)

	cpdconv10 = Conv2DTranspose(2, kernel_size=filters,
	padding='same', activation='elu')(cpdconv9)

	# Last Layer (12)
	cpdconv11 = Conv2DTranspose(1, kernel_size=filters,
	padding='same', activation='linear')(cpdconv10)

	model = Model(inputs=[input_layer1], outputs=[cpdconv11])
	return model
