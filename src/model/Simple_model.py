from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.initializers import glorot_uniform

def network(input_shape = (240, 320, 1), filter_size= (3,3), filters_num = (16, 32, 64, 128), s = 1):
    '''
    Build the simple network.
    Network inputs are 2 images (template and rotated_pattern).
    Input => ( CONV => BN => RELU => POOL ) => FC => BN => RELU => regression => Output
    Network output is a scalar rotation angle (degree of rotation compare to the template)

    Inputs:
        input_shape: (height, width, depth) of the input image.
        filter_size: 
        filters_num:
        s: stride of Convolute layer.

    Output:
        model: a keras model
    '''

    X_input = Input(input_shape)
    chanDim = -1

    # loop over the number of filters
    for i, f in enumerate(filters_num):
		# if this is the first CONV layer then set the input
        if i == 0:
            x = X_input
        
        x = Conv2D(f, filter_size, strides = (s,s), padding="same", kernel_initializer = glorot_uniform())(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # flatten the volume, then FC => BN => RELU
    x = Flatten()(x)
    x = Dense(16)(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Activation("relu")(x)

    # regression
    x = Dense(1, activation="linear")(x)

    # construct the CNN
    model = Model(X_input, x)

    # return the CNN
    return model