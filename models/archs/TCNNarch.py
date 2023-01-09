import numpy as np
import tensorflow.keras.layers as tfkl
from tensorflow.keras.layers import TimeDistributed, Cropping2D, Add, Conv1D, Conv2D, \
                                    Conv2DTranspose, Input, MaxPooling2D, Reshape, \
                                    UpSampling2D, ZeroPadding2D
from tensorflow.keras.models import Model


def build_model(input_shape, af='elu'):
    ks = (4,4)
    pool = (2,2)

    inputs = Input(shape=input_shape)
    settings = dict(kernel_size=(4, 4), padding='same', activation=af)

    x = TimeDistributed(Conv2D(9, **settings))(inputs)
    x = TimeDistributed(Conv2D(9, **settings))(x)
    x = tfkl.Add()([inputs, x])
    print(str(x.shape))

    conv2d2_1 = TimeDistributed(Conv2D(16, kernel_size=ks,           padding='same', activation=af))(x)
    conv2d2 = TimeDistributed(Conv2D(16, kernel_size=ks,           padding='same', activation=af))(conv2d2_1)
    conv2d2 = tfkl.Add()([conv2d2_1, conv2d2])
    print(str(conv2d2.shape))

    conv2d3_1 = TimeDistributed(Conv2D(32, kernel_size=ks,           padding='same', activation=af))(conv2d2)
    conv2d3 = TimeDistributed(Conv2D(32, kernel_size=ks,           padding='same', activation=af))(conv2d3_1)
    conv2d3 = tfkl.Add()([conv2d3_1, conv2d3])
    pool1 = TimeDistributed(MaxPooling2D(pool_size=pool))(conv2d3)
    print(str(pool1.shape))

    conv2d4_1 = TimeDistributed(Conv2D(64, kernel_size=ks,           padding='same', activation=af))(pool1)
    conv2d4 = TimeDistributed(Conv2D(64, kernel_size=ks,           padding='same', activation=af))(conv2d4_1)
    conv2d4 = tfkl.Add()([conv2d4_1, conv2d4])
    pool2 = TimeDistributed(MaxPooling2D(pool_size=pool))(conv2d4)
    print(str(pool2.shape))

    conv2d5_1 = TimeDistributed(Conv2D(128, kernel_size=ks,           padding='same', activation=af))(pool2)
    conv2d5 = TimeDistributed(Conv2D(128, kernel_size=ks,           padding='same', activation=af))(conv2d5_1)
    conv2d5 = tfkl.Add()([conv2d5_1, conv2d5])
    pool3 = TimeDistributed(MaxPooling2D(pool_size=pool))(conv2d5)
    crop5 = TimeDistributed(Cropping2D(cropping=((3,1),(2,0)), data_format="channels_last"))(pool3)
    print(str(crop5.shape))

    conv2d6_1 = TimeDistributed(Conv2D(256, kernel_size=ks,           padding='same', activation=af))(crop5)
    conv2d6 = TimeDistributed(Conv2D(256, kernel_size=ks,           padding='same', activation=af))(conv2d6_1)
    conv2d6 = tfkl.Add()([conv2d6_1, conv2d6])
    pool4 = TimeDistributed(MaxPooling2D(pool_size=pool))(conv2d6)
    print(str(pool4.shape))

    conv2d7_1 = TimeDistributed(Conv2D(512, kernel_size=ks,           padding='same', activation=af))(pool4)
    conv2d7 = TimeDistributed(Conv2D(512, kernel_size=ks,           padding='same', activation=af))(conv2d7_1)
    conv2d7 = tfkl.Add()([conv2d7_1, conv2d7])
    pool5 = TimeDistributed(MaxPooling2D(pool_size=pool))(conv2d7)
    crop7 = TimeDistributed(Cropping2D(cropping=((0,0),(0,0)), data_format="channels_last"))(pool5)
    print(str(crop7.shape))

    conv2d8_1 = TimeDistributed(Conv2D(1024, kernel_size=ks,           padding='same', activation=af))(crop7)
    conv2d8 = TimeDistributed(Conv2D(1024, kernel_size=ks,           padding='same', activation=af))(conv2d8_1)
    conv2d8 = tfkl.Add()([conv2d8_1, conv2d8])
    pool6 = TimeDistributed(MaxPooling2D(pool_size=pool))(conv2d8)
    print(str(pool6.shape))

    tvector = Reshape((3,1024))(pool6)
    print('Encoder output {}'.format(tvector.shape))

    fts = int(tvector.shape[2])
    fts2 = fts*2
    print('Filter x2: {}'.format(fts2))

    print(str(tvector.shape))
    conv1d1_1 = Conv1D(fts2, kernel_size=2, padding='causal', dilation_rate=1, activation=af)(tvector)
    conv1d1 = Conv1D(fts2, kernel_size=2, padding='causal', dilation_rate=1, activation=af)(conv1d1_1)
    conv1d1 = Conv1D(fts2, kernel_size=2, padding='causal', dilation_rate=1, activation=af)(conv1d1)
    conv1d1 = tfkl.Add()([conv1d1_1, conv1d1])
    print(str(conv1d1.shape))

    conv1d2_1 = Conv1D(fts2, kernel_size=2, padding='causal', dilation_rate=1, activation=af)(conv1d1)
    conv1d2 = Conv1D(fts2, kernel_size=2, padding='causal', dilation_rate=2, activation=af)(conv1d2_1)
    conv1d2 = Conv1D(fts2, kernel_size=2, padding='causal', dilation_rate=1, activation=af)(conv1d2)
    conv1d2 = tfkl.Add()([conv1d2_1, conv1d2])
    print(str(conv1d2.shape))

    conv1d3_1 = Conv1D(fts2, kernel_size=2, padding='causal', dilation_rate=1, activation=af)(conv1d2)
    conv1d3 = Conv1D(fts2, kernel_size=2, padding='causal', dilation_rate=2, activation=af)(conv1d3_1)
    conv1d3 = Conv1D(fts2, kernel_size=2, padding='causal', dilation_rate=1, activation=af)(conv1d3)
    conv1d3 = tfkl.Add()([conv1d3_1, conv1d3])
    print(str(conv1d3.shape))

    conv1d4_1 = Conv1D(fts2, kernel_size=2, padding='causal', dilation_rate=1, activation=af)(conv1d3)
    conv1d4 = Conv1D(fts2, kernel_size=2, padding='causal', dilation_rate=2, activation=af)(conv1d4_1)
    conv1d4 = Conv1D(fts2, kernel_size=2, padding='causal', dilation_rate=1, activation=af)(conv1d4)
    conv1d4 = tfkl.Add()([conv1d4_1, conv1d4])
    print(str(conv1d4.shape))

    conv1d5_1 = Conv1D(fts2, kernel_size=2, padding='causal', dilation_rate=1, activation=af)(conv1d4)
    conv1d5 = Conv1D(fts2, kernel_size=2, padding='causal', dilation_rate=12, activation=af)(conv1d5_1)
    conv1d5 = Conv1D(fts2, kernel_size=2, padding='causal', dilation_rate=1, activation=af)(conv1d5)
    conv1d5 = tfkl.Add()([conv1d5_1, conv1d5])
    print(str(conv1d5.shape))

    conv1d2_1 = Conv1D(fts2, kernel_size=2, padding='causal', dilation_rate=1, activation=af)(conv1d5)
    conv1d2 = Conv1D(fts2, kernel_size=2, padding='causal', dilation_rate=32, activation=af)(conv1d2_1)
    conv1d2 = Conv1D(fts2, kernel_size=2, padding='causal', dilation_rate=1, activation=af)(conv1d2)
    conv1d2 = tfkl.Add()([conv1d2_1, conv1d2])
    print(str(conv1d2.shape))

    ttensor = Reshape((3,1,1,2048))(conv1d2)
    print(str(ttensor.shape))

    print(str(ttensor.shape))
    deconv2d1_1 = TimeDistributed(Conv2DTranspose(1024, kernel_size=ks,
            padding='same', activation=af))(ttensor)
    deconv2d1 = TimeDistributed(Conv2DTranspose(1024, kernel_size=ks,
            padding='same', activation=af))(deconv2d1_1)
    up0 = TimeDistributed(UpSampling2D(size=pool))(deconv2d1)
    print(str(up0.shape))

    deconv2d2_1 = TimeDistributed(Conv2DTranspose(512, kernel_size=ks,
            padding='same', activation=af))(up0)
    deconv2d2 = TimeDistributed(Conv2DTranspose(512, kernel_size=ks,
            padding='same', activation=af))(deconv2d2_1)
    deconv2d2 = tfkl.Add()([deconv2d2, deconv2d2_1])
    up1 = TimeDistributed(UpSampling2D(size=pool))(deconv2d2)
    decrop7 = TimeDistributed(Cropping2D(cropping=((0,0),(0,0)), data_format="channels_last"))(up1)
    print(str(decrop7.shape))

    deconv2d3_1 = TimeDistributed(Conv2DTranspose(256, kernel_size=ks,
            padding='same', activation=af))(decrop7)
    deconv2d3 = TimeDistributed(Conv2DTranspose(256, kernel_size=ks,
            padding='same', activation=af))(deconv2d3_1)
    deconv2d3 = tfkl.Add()([deconv2d3_1, deconv2d3])
    up2 = TimeDistributed(UpSampling2D(size=pool))(deconv2d3)
    dcrop6 = TimeDistributed(Cropping2D(cropping=((0,0),(0,0)), data_format="channels_last"))(up2)
    print(str(dcrop6.shape))

    deconv2d4_1 = TimeDistributed(Conv2DTranspose(128, kernel_size=ks,
            padding='same', activation=af))(dcrop6)
    deconv2d4 = TimeDistributed(Conv2DTranspose(128, kernel_size=ks,
            padding='same', activation=af))(deconv2d4_1)
    deconv2d4 = tfkl.Add()([deconv2d4_1, deconv2d4])
    up3 = TimeDistributed(UpSampling2D(size=(4,4)))(deconv2d4)
    dcrop7 = TimeDistributed(Cropping2D(cropping=((3,4),(6,6)), data_format="channels_last"))(up3)
    print(str(dcrop7.shape))

    deconv2d5_1 = TimeDistributed(Conv2DTranspose(64, kernel_size=ks,
            padding='same', activation=af))(dcrop7)
    deconv2d5 = TimeDistributed(Conv2DTranspose(64, kernel_size=ks,
            padding='same', activation=af))(deconv2d5_1)
    deconv2d5 = tfkl.Add()([deconv2d5_1, deconv2d5])
    up4 = TimeDistributed(UpSampling2D(size=pool))(deconv2d5)
    zeropad5 = TimeDistributed(ZeroPadding2D(padding=((0,0),(1,0))))(up4)
    print(str(zeropad5.shape))

    deconv2d6_1 = TimeDistributed(Conv2DTranspose(32, kernel_size=ks,
            padding='same', activation=af))(zeropad5)
    deconv2d6 = TimeDistributed(Conv2DTranspose(32, kernel_size=ks,
            padding='same', activation=af))(deconv2d6_1)
    deconv2d6 = tfkl.Add()([deconv2d6_1, deconv2d6])
    up5 = TimeDistributed(UpSampling2D(size=pool))(deconv2d6)
    zeropad6 = TimeDistributed(ZeroPadding2D(padding=((1,0),(0,0))))(up5)
    print(str(zeropad6.shape))

    deconv2d7_1 = TimeDistributed(Conv2DTranspose(16, kernel_size=ks,
            padding='same', activation=af))(zeropad6)
    deconv2d7 = TimeDistributed(Conv2DTranspose(16, kernel_size=ks,
            padding='same', activation=af))(deconv2d7_1)
    deconv2d7 = tfkl.Add()([deconv2d7_1, deconv2d7])
    print(str(deconv2d7.shape))

    deconv2d8_1 = TimeDistributed(Conv2DTranspose(9, kernel_size=ks,
            padding='same', activation=af))(deconv2d7)
    deconv2d8 = TimeDistributed(Conv2DTranspose(9, kernel_size=ks,
            padding='same', activation=af))(deconv2d8_1)
    deconv2d8 = tfkl.Add()([deconv2d8_1, deconv2d8])
    print(str(deconv2d8.shape))

    deconv2d9 = TimeDistributed(Conv2DTranspose(4, kernel_size=ks,
            padding='same', activation=af))(deconv2d8)
    print(str(deconv2d9.shape))

    deconv2d10 = TimeDistributed(Conv2DTranspose(2, kernel_size=ks,
            padding='same', activation=af))(deconv2d9)
    print(str(deconv2d10.shape))

    deconv2d11 = TimeDistributed(Conv2DTranspose(1, kernel_size=ks,
            padding='same', activation='linear'))(deconv2d10)

    print(str(deconv2d11.shape))

    model = Model(inputs=[inputs], outputs=[deconv2d11])

    return model

if __name__ == '__main__':

    #Xrs = np.zeros((10,3,101,82,9))
    #print('Input shape: ',Xrs.shape)
    #tcnn_model = buildTCNN(Xrs)
    tcnn_model = build_model((3, 101, 82, 9))
    tcnn_model.summary()
