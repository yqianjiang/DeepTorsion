from model.STN import STN
from model.Simple_model import network
from utils.preprocess import make_subset
from utils.DataGen import DataGenerator

from keras import losses, optimizers, metrics
from keras.callbacks import TensorBoard, ModelCheckpoint

import math
import numpy as np

import os

import keras.backend as K
import tensorflow as tf
#from utils.noise_function import noise
#import threading

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def cc(y_true, y_pred):
    template = y_true
    resample = y_pred

    # mask out "0"
    mask_t = tf.math.greater(template, 0)
    mask_r = tf.math.greater(resample, 0)
    mask = tf.math.logical_and(mask_t, mask_r)
    v_t = tf.boolean_mask(template, mask)
    v_r = tf.boolean_mask(resample, mask)

    # calculate cross-correlation
    zm_t = (v_t - K.mean(v_t)) / K.std(v_t)
    zm_r = (v_r - K.mean(v_r)) / K.std(v_r)
    output = K.mean(zm_t*zm_r)

    # adjust to "loss": lower number for more similar image (original: -1..1, after become 0..2)
    output *= -1
    output += 1

    return output



def fit_model_dir(data_dir, n_rotate=8, h=60, w=360, batch_size=16, epochs=50, stn=True,
                  add_noise=False, condition="", degree_sampling="uniform", model_save_dir=None, max_degree=15):
    '''
    Train the model.

    Args:
        data_dir: directory contains images to train.
        N: number of training images.

    Network inputs:
        X_train: (N*n_rotate, height, width, depth). 2 iris pattern (template and rotated pattern)
                --(N*n_rotate, h, w, 2)
        Y_train: (N*n_rotate ). Degree of rotation.

    Output:
        model: model that can use to predict new data.
    '''
    log_dir = './logs/' + condition
    train_dir, val_dir = make_subset(data_dir)
    N = len(train_dir)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    # tensorborad settings
    tb_callback = TensorBoard(log_dir=log_dir,
                              histogram_freq=0,
                              batch_size=batch_size,
                              write_graph=True,
                              write_grads=True,
                              write_images=True,
                              embeddings_freq=0,
                              embeddings_layer_names=None,
                              embeddings_metadata=None,
                              update_freq='epoch')

    if model_save_dir is not None:
        file_path = os.path.join(model_save_dir, condition+".h5")
        checkpoint = ModelCheckpoint(
            file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks = [tb_callback, checkpoint]
    else:
        callbacks = [tb_callback]

    # compile loss function and optimizer.
    if stn:
        model = STN((h, w, 2))
        adam = optimizers.Adam(lr = 0.001)
        model.compile(loss={'resample': cc, 'theta': 'mean_squared_error'},
                    loss_weights={'resample': 1., 'theta': .02},
                    optimizer=adam, metrics={'resample': cc, 'theta': rmse})
    else:
        model = network((h, w, 2))
        adam = optimizers.Adam(lr = 0.001)
        model.compile(loss='mean_squared_error',
                    optimizer=adam, metrics=[rmse])


    data_gen = DataGenerator(horizontal_flip=False, stn=stn)

    #train_generator = data_gen.flow(X_train[0:split], Y_train[0:split], batch_size = batch_size)
    train_generator = data_gen.gen_flow(train_dir, batch_size, n_rotate, h=h, w=w,
                                        degree_sampling=degree_sampling, add_noise=add_noise, seed=0, max_degree=max_degree)
    val_generator = data_gen.gen_flow(
        val_dir, batch_size, n_rotate, h=h, w=w, degree_sampling=degree_sampling, add_noise=add_noise, seed=0, max_degree=max_degree)

    # fit model
    model.fit_generator(train_generator,
                        validation_data=val_generator,
                        verbose=0,
                        steps_per_epoch=math.ceil(N/batch_size),
                        epochs=epochs,
                        validation_steps=math.ceil(len(val_dir) / batch_size),
                        workers=4,
                        callbacks=callbacks)

    return model
