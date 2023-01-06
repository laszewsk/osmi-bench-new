from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, TimeDistributed


def build_model(input_shape, af='elu'):

    inputs = Input(shape=input_shape, name='input')
    x = TimeDistributed(Conv2D(1024, (3, 3), padding='same'))(inputs)
    x = TimeDistributed(Conv2D(1024, (3, 3), padding='same'))(x)
    x = TimeDistributed(Conv2D(1024, (3, 3), padding='same'))(x)
    x = TimeDistributed(Conv2D(1024, (3, 3), padding='same'))(x)
    x = TimeDistributed(Conv2D(1024, (3, 3), padding='same'))(x)
    x = TimeDistributed(Conv2D(1024, (3, 3), padding='same'))(x)
    x = TimeDistributed(Conv2D(1024, (3, 3), padding='same'))(x)
    x = TimeDistributed(Conv2D(1024, (3, 3), padding='same'))(x)
    x = TimeDistributed(Conv2D(1024, (3, 3), padding='same'))(x)
    x = TimeDistributed(Conv2D(1024, (3, 3), padding='same'))(x)
    x = TimeDistributed(Conv2D(1024, (3, 3), padding='same'))(x)
    x = TimeDistributed(Conv2D(1024, (3, 3), padding='same'))(x)
    x = TimeDistributed(Conv2D(1024, (3, 3), padding='same'))(x)
    x = TimeDistributed(Conv2D(1024, (3, 3), padding='same'))(x)
    x = TimeDistributed(Conv2D(1024, (3, 3), padding='same'))(x)
    x = TimeDistributed(Conv2D(1024, (3, 3), padding='same'))(x)
    x = TimeDistributed(Conv2D(1024, (3, 3), padding='same'))(x)
    x = TimeDistributed(Conv2D(1024, (3, 3), padding='same'))(x)
    x = TimeDistributed(Conv2D(1024, (3, 3), padding='same'))(x)
    x = TimeDistributed(Conv2D(1024, (3, 3), padding='same'))(x)
    x = TimeDistributed(Conv2D(1024, (3, 3), padding='same'))(x)
    x = TimeDistributed(Conv2D(1024, (3, 3), padding='same'))(x)
    x = TimeDistributed(Conv2D(1024, (3, 3), padding='same'))(x)
    outputs = TimeDistributed(Conv2D(1, (3, 3), padding='same'))(x)

    return Model(inputs=[inputs], outputs=[outputs])
