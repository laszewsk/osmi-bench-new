import numpy as np
import tensorflow.keras.layers as tfkl
from tensorflow.keras.layers import TimeDistributed, Cropping2D, Add, Conv1D, Conv2D, \
                                    Conv2DTranspose, Input, MaxPooling2D, Reshape, \
                                    UpSampling2D, ZeroPadding2D
from tensorflow.keras.models import Model

def resblock(inputs, units, settings):
    x1 = TimeDistributed(Conv2D(units, **settings))(inputs)
    x2 = TimeDistributed(Conv2D(units, **settings))(x1)
    return tfkl.Add()([x1, x2])

def build_model(input_shape, af='elu'):
    ks = (4,4)
    pool = (2,2)

    inputs = Input(shape=input_shape)
    settings = dict(kernel_size=(4, 4), padding='same', activation=af)
    print(settings)

    x = resblock(inputs, 9, settings)
    #x = TimeDistributed(Conv2D(9, **settings))(inputs)
    #x = TimeDistributed(Conv2D(9, **settings))(x)
    #x = tfkl.Add()([inputs, x])

    x1 = TimeDistributed(Conv2D(16, **settings))(x)
    x2 = TimeDistributed(Conv2D(16, **settings))(x1)
    x = tfkl.Add()([x1, x2])

    x1 = TimeDistributed(Conv2D(32, **settings))(x)
    x2 = TimeDistributed(Conv2D(32, **settings))(x1)
    x = tfkl.Add()([x1, x2])
    x = TimeDistributed(MaxPooling2D(pool_size=pool))(x)

    x1 = TimeDistributed(Conv2D(64, **settings))(x)
    x2 = TimeDistributed(Conv2D(64, **settings))(x1)
    x = tfkl.Add()([x1, x2])
    x = TimeDistributed(MaxPooling2D(pool_size=pool))(x)

    x1 = TimeDistributed(Conv2D(128, **settings))(x)
    x2 = TimeDistributed(Conv2D(128, **settings))(x1)
    x = tfkl.Add()([x1, x2])
    x = TimeDistributed(MaxPooling2D(pool_size=pool))(x)
    x = TimeDistributed(Cropping2D(cropping=((3, 1), (2, 0)), \
                                   data_format="channels_last"))(x)

    x1 = TimeDistributed(Conv2D(256, **settings))(x)
    x2 = TimeDistributed(Conv2D(256, **settings))(x1)
    x = tfkl.Add()([x1, x2])
    x = TimeDistributed(MaxPooling2D(pool_size=pool))(x)

    x1 = TimeDistributed(Conv2D(512, **settings))(x)
    x2 = TimeDistributed(Conv2D(512, **settings))(x1)
    x = tfkl.Add()([x1, x2])
    x = TimeDistributed(MaxPooling2D(pool_size=pool))(x)
    x = TimeDistributed(Cropping2D(cropping=((0, 0), (0, 0)), \
                                   data_format="channels_last"))(x)

    x1 = TimeDistributed(Conv2D(1024, **settings))(x)
    x2 = TimeDistributed(Conv2D(1024, **settings))(x1)
    x = tfkl.Add()([x1, x2])
    x = TimeDistributed(MaxPooling2D(pool_size=pool))(x)

    x = Reshape((3, 1024))(x)
    print('Encoder output {}'.format(x.shape))

    fts = int(x.shape[2])
    fts2 = fts*2

    settings2 = dict(kernel_size=2, padding='causal', activation=af)

    conv1d1_1 = Conv1D(fts2, dilation_rate=1, **settings2)(x)
    conv1d1 = Conv1D(fts2, dilation_rate=1, **settings2)(conv1d1_1)
    conv1d1 = Conv1D(fts2, dilation_rate=1, **settings2)(conv1d1)
    conv1d1 = tfkl.Add()([conv1d1_1, conv1d1])

    conv1d2_1 = Conv1D(fts2, dilation_rate=1, **settings2)(conv1d1)
    conv1d2 = Conv1D(fts2, dilation_rate=2, **settings2)(conv1d2_1)
    conv1d2 = Conv1D(fts2, dilation_rate=1, **settings2)(conv1d2)
    conv1d2 = tfkl.Add()([conv1d2_1, conv1d2])

    conv1d3_1 = Conv1D(fts2, dilation_rate=1, **settings2)(conv1d2)
    conv1d3 = Conv1D(fts2, dilation_rate=2, **settings2)(conv1d3_1)
    conv1d3 = Conv1D(fts2, dilation_rate=1, **settings2)(conv1d3)
    conv1d3 = tfkl.Add()([conv1d3_1, conv1d3])

    conv1d4_1 = Conv1D(fts2, dilation_rate=1, **settings2)(conv1d3)
    conv1d4 = Conv1D(fts2, dilation_rate=2, **settings2)(conv1d4_1)
    conv1d4 = Conv1D(fts2, dilation_rate=1, **settings2)(conv1d4)
    conv1d4 = tfkl.Add()([conv1d4_1, conv1d4])

    conv1d5_1 = Conv1D(fts2, dilation_rate=1, **settings2)(conv1d4)
    conv1d5 = Conv1D(fts2, dilation_rate=12, **settings2)(conv1d5_1)
    conv1d5 = Conv1D(fts2, dilation_rate=1, **settings2)(conv1d5)
    conv1d5 = tfkl.Add()([conv1d5_1, conv1d5])

    conv1d2_1 = Conv1D(fts2, dilation_rate=1, **settings2)(conv1d5)
    conv1d2 = Conv1D(fts2, dilation_rate=32, **settings2)(conv1d2_1)
    conv1d2 = Conv1D(fts2, dilation_rate=1, **settings2)(conv1d2)
    conv1d2 = tfkl.Add()([conv1d2_1, conv1d2])

    ttensor = Reshape((3, 1, 1, 2048))(conv1d2)

    deconv2d1_1 = TimeDistributed(Conv2DTranspose(1024, **settings))(ttensor)
    deconv2d1 = TimeDistributed(Conv2DTranspose(1024, **settings))(deconv2d1_1)
    up0 = TimeDistributed(UpSampling2D(size=pool))(deconv2d1)

    deconv2d2_1 = TimeDistributed(Conv2DTranspose(512, **settings))(up0)
    deconv2d2 = TimeDistributed(Conv2DTranspose(512, **settings))(deconv2d2_1)
    deconv2d2 = tfkl.Add()([deconv2d2, deconv2d2_1])
    up1 = TimeDistributed(UpSampling2D(size=pool))(deconv2d2)
    decrop7 = TimeDistributed(Cropping2D(cropping=((0,0),(0,0)), data_format="channels_last"))(up1)

    deconv2d3_1 = TimeDistributed(Conv2DTranspose(256, **settings))(decrop7)
    deconv2d3 = TimeDistributed(Conv2DTranspose(256, **settings))(deconv2d3_1)
    deconv2d3 = tfkl.Add()([deconv2d3_1, deconv2d3])
    up2 = TimeDistributed(UpSampling2D(size=pool))(deconv2d3)
    dcrop6 = TimeDistributed(Cropping2D(cropping=((0,0),(0,0)), data_format="channels_last"))(up2)

    deconv2d4_1 = TimeDistributed(Conv2DTranspose(128, **settings))(dcrop6)
    deconv2d4 = TimeDistributed(Conv2DTranspose(128, **settings))(deconv2d4_1)
    deconv2d4 = tfkl.Add()([deconv2d4_1, deconv2d4])
    up3 = TimeDistributed(UpSampling2D(size=(4,4)))(deconv2d4)
    dcrop7 = TimeDistributed(Cropping2D(cropping=((3,4),(6,6)), data_format="channels_last"))(up3)

    deconv2d5_1 = TimeDistributed(Conv2DTranspose(64, **settings))(dcrop7)
    deconv2d5 = TimeDistributed(Conv2DTranspose(64, **settings))(deconv2d5_1)
    deconv2d5 = tfkl.Add()([deconv2d5_1, deconv2d5])
    up4 = TimeDistributed(UpSampling2D(size=pool))(deconv2d5)
    zeropad5 = TimeDistributed(ZeroPadding2D(padding=((0,0),(1,0))))(up4)

    deconv2d6_1 = TimeDistributed(Conv2DTranspose(32, **settings))(zeropad5)
    deconv2d6 = TimeDistributed(Conv2DTranspose(32, **settings))(deconv2d6_1)
    deconv2d6 = tfkl.Add()([deconv2d6_1, deconv2d6])
    up5 = TimeDistributed(UpSampling2D(size=pool))(deconv2d6)
    zeropad6 = TimeDistributed(ZeroPadding2D(padding=((1,0),(0,0))))(up5)

    deconv2d7_1 = TimeDistributed(Conv2DTranspose(16, **settings))(zeropad6)
    deconv2d7 = TimeDistributed(Conv2DTranspose(16, **settings))(deconv2d7_1)
    deconv2d7 = tfkl.Add()([deconv2d7_1, deconv2d7])

    deconv2d8_1 = TimeDistributed(Conv2DTranspose(9, **settings))(deconv2d7)
    deconv2d8 = TimeDistributed(Conv2DTranspose(9, **settings))(deconv2d8_1)
    deconv2d8 = tfkl.Add()([deconv2d8_1, deconv2d8])

    deconv2d9 = TimeDistributed(Conv2DTranspose(4, **settings))(deconv2d8)

    deconv2d10 = TimeDistributed(Conv2DTranspose(2, **settings))(deconv2d9)

    settings['activation'] = 'linear'
    deconv2d11 = TimeDistributed(Conv2DTranspose(1, **settings))(deconv2d10)

    return Model(inputs=[inputs], outputs=[deconv2d11])


if __name__ == '__main__':
    tcnn_model = build_model((3, 101, 82, 9))
    tcnn_model.summary()
