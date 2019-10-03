from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.layers import Flatten, Dense, Lambda
from keras.models import Model
from keras.initializers import glorot_uniform
from .transformer import BilinearInterpolation
import numpy as np
from keras import backend as K


def STN(input_shape=(60, 720, 2), filter_size=(3, 3), filters_num=(16, 32, 64, 128), s=1):
    '''
    Build the STN.
    2 inputs for the network:
        X_input: shape(60, 720, 2), chan0 = template, chan1 = rotate, for locNet to predict rotation angle.
        input_image: shape(60, 720, 1), rotate image, for STN to apply transform.
    2 outputs for the network:
        theta: rotation angle predicted by locNet, for RMSE loss.
        resample: shape(60, 720, 1), resample image transformed from input_image, for cross-correlation loss with template.

    Inputs:
        input_shape: (height, width, depth) of the input image.
        filter_size: 
        filters_num:
        s: stride of Convolute layer.

    Output:
        model: a keras model
    '''
    h, w, _ = input_shape
    X_input = Input(shape = input_shape, name='X')
    input_image = Input(shape = (h, w, 1), name='rotate')

    theta = locNet(X_input, filter_size, filters_num, s)
    M = Lambda(make_matrix)(theta)    
    resample = BilinearInterpolation((h, w), name='resample')([input_image, M])

    # construct the CNN
    model = Model(inputs=[X_input, input_image], outputs = [theta, resample])

    # return the CNN
    return model

def make_matrix(x):
    '''
    input:
        x: rotation angle in pixel unit. Tensor("theta/BiasAdd:0", shape=(256=16*16, 1), dtype=float32)
    output:
        M: transform matrix for the STN.  shape = (256, 6)
            = [[1, 0, dx],
               [0, 1, 0]], where dx = 2 * pixel / width
    '''
    import tensorflow as tf
    v1 = tf.constant([0., 0., 1., 0., 0., 0.], dtype='float32')
    v2 = tf.constant([1., 0., 0., 0., 1., 0.], dtype='float32')
    v1x = x * 2.0 / 720 * v1
    M = v1x + v2
    return M


def locNet(x, filter_size=(3, 3), filters_num=(16, 32, 64, 128), s=1, chanDim=-1):
    # loop over the number of filters
    for f in filters_num:
        x = Conv2D(f, filter_size, strides=(s, s), padding="same",
                   kernel_initializer=glorot_uniform())(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # flatten the volume, then FC => BN => RELU
    x = Flatten()(x)
    x = Dense(256)(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Activation("relu")(x)
    x = Dense(16)(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Activation("relu")(x)

    # regression
    x = Dense(1, activation="linear", name="theta")(x)

    return x
