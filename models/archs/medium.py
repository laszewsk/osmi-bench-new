from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPooling2D, Dense, Flatten, Cropping2D
from tensorflow.keras.layers import Reshape, Conv2DTranspose, ZeroPadding2D, Conv2D


def build_model(input_shape, filters=(3, 3), actfn='elu', padding='same'):

	input_layer1 = Input(shape=input_shape, name='input')
	pool = (2,2)

	x = Conv2D(32, kernel_size=filters, padding=padding, activation=actfn)(input_layer1)

	x = Conv2D(64, kernel_size=filters, padding=padding, activation=actfn)(x)
	x = MaxPooling2D(pool_size=pool)(x)

	x = Conv2D(128, kernel_size=filters, padding=padding, activation=actfn)(x)
	x = MaxPooling2D(pool_size=pool)(x)

	x = Conv2D(256, kernel_size=filters, padding=padding, activation=actfn)(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Cropping2D(cropping=((2, 2), (1, 1)))(x)

	x = Conv2D(512, kernel_size=filters, padding=padding, activation=actfn)(x)
	x = MaxPooling2D(pool_size=pool)(x)

	x = Conv2D(1024, kernel_size=filters, padding=padding, activation=actfn)(x)
	x = MaxPooling2D(pool_size=(4, 4))(x)

	flat_units = int(x.shape[3])

	x = Flatten()(x)

	# Dense fully connected regression
	x = Dense(flat_units, activation=actfn)(x)

	# Reshape before decoding
	x = Reshape((1, 1, flat_units))(x)
	x = Conv2DTranspose(flat_units, kernel_size=filters, strides=(1, 1), padding=padding, activation=actfn)(x)
	x = Conv2DTranspose(512, kernel_size=filters, strides=(2, 2), padding=padding, activation=actfn)(x)
	x = Conv2DTranspose(256, kernel_size=filters, strides=(2, 2), padding=padding, activation=actfn)(x)
	x = Conv2DTranspose(128, kernel_size=filters, strides=(2, 2), padding=padding, activation=actfn)(x)
	x = Conv2DTranspose(64, kernel_size=filters, strides=(4, 4), padding=padding, activation=actfn)(x)
	x = Cropping2D(cropping=((4, 3),(6, 6)))(x)

	x = Conv2DTranspose(32, kernel_size=filters, strides=(2, 2), padding=padding, activation=actfn)(x)
	x = ZeroPadding2D(padding=((0, 0), (1, 0)))(x)

	x = Conv2DTranspose(16, kernel_size=filters, strides=(2, 2), padding=padding, activation=actfn)(x)
	x = ZeroPadding2D(padding=((1,0),(0,0)))(x)

	x = Conv2DTranspose(8, kernel_size=filters, strides=(1, 1), padding=padding, activation=actfn)(x)
	x = Conv2DTranspose(4, kernel_size=filters, padding=padding, activation=actfn)(x)
	x = Conv2DTranspose(2, kernel_size=filters, padding=padding, activation=actfn)(x)

	outputs = Conv2DTranspose(1, kernel_size=filters, padding=padding, activation='linear')(x)

	return Model(inputs=[input_layer1], outputs=[outputs])

