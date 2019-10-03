from keras.models import load_model
from model.transformer import BilinearInterpolation
import keras.backend as K
import tensorflow as tf

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


def load_stn(model_path):
    model = load_model(model_path, custom_objects={
                       "rmse": rmse, "cc": cc, "BilinearInterpolation": BilinearInterpolation})
    return model
