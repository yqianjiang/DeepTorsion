from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import warnings
import scipy
from scipy import linalg
from .my_iterator import MyIterator

class DataGenerator(ImageDataGenerator):
    def __init__(self, stn=False,
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                zca_epsilon=1e-06,
                rotation_range=0,
                width_shift_range=0.0,
                height_shift_range=0.0,
                brightness_range=None,
                shear_range=0.0,
                zoom_range=0.0,
                channel_shift_range=0.0,
                fill_mode='nearest',
                cval=0.0,
                horizontal_flip=False,
                vertical_flip=False,
                rescale=None,
                preprocessing_function=None,
                data_format=None,
                validation_split=0.0,
                dtype=None):
        self.stn = stn
        super(DataGenerator, self).__init__(
                featurewise_center=featurewise_center,
                samplewise_center=samplewise_center,
                featurewise_std_normalization=featurewise_std_normalization,
                samplewise_std_normalization=samplewise_std_normalization,
                zca_whitening=zca_whitening,
                zca_epsilon=zca_epsilon,
                rotation_range=rotation_range,
                width_shift_range=width_shift_range,
                height_shift_range=height_shift_range,
                brightness_range=brightness_range,
                shear_range=shear_range,
                zoom_range=zoom_range,
                channel_shift_range=channel_shift_range,
                fill_mode=fill_mode,
                cval=cval,
                horizontal_flip=horizontal_flip,
                vertical_flip=vertical_flip,
                rescale=rescale,
                preprocessing_function=preprocessing_function,
                data_format=data_format,
                validation_split=validation_split,
                dtype=dtype)

    def gen_flow(self, dir, batch_size, n_rotate, h, w, degree_sampling, add_noise, shuffle = True, seed = None, max_degree = 15):
        return MyIterator(dir, batch_size, n_rotate, h, w, degree_sampling, add_noise, shuffle, seed, max_degree, stn=self.stn)
