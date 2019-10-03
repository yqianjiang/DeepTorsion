import numpy as np
import cv2
from .noise_function import noise


def preprocess_batch(template_batch, n_rotate = 8, resolution=0.02, max_degree=15, degree_sampling="uniform", add_noise = False, h = 60, w = 360):
    '''
    Generate rotate image from a batch of templates.

    Args:
        template_batch: size(batch_size, h, w)
        n_rotate: int. how many rotated image to generate
        max_degree: scalar. degree range for rotation
    return:
        batch_x: size(batch_size * n_rotate, h, w, 2)
        batch_y: size (batchsize*n_rotate, )  ----rotation degree in pixel unit.
    '''
    batch_size = len(template_batch)
    pix_resolu = 360/w

    # Initialize numpy array
    batch_x = np.zeros((batch_size, n_rotate, h, w, 2))
    batch_y = np.zeros((batch_size, n_rotate))

    for i, image in enumerate(template_batch):
        degree_seq = get_degreee_seq(n_rotate, max_degree, degree_sampling=degree_sampling)
        rotate_batch = get_rotate(image, degree_seq, h, w)
        batch_x[i] = rotate_batch
        batch_y[i] = degree_seq/pix_resolu

    batch_x = batch_x.reshape((batch_size * n_rotate, h, w, 2))
    if add_noise:
        batch_x = preprocess(batch_x)

    return batch_x, batch_y.flatten()


def make_subset(img_dir):
    '''
    Split training set into training set and validation set.
    Args:
        img_dir: a directory like  "training/polar" that contains all the template images.
    Returns:
        train: a list of path as training image.
        val: a list of path as validation image.
    '''
    img_dirs = []
    for name in os.listdir(img_dir):
        img_dirs.append(img_dir + name)
    
    N = len(img_dirs)
    split = int(N * 0.8)
    train = img_dirs[0:split]
    val = img_dirs[split:N]
    return train, val



def preprocess(x_batch):
    '''
    add noise to the image.
    '''
    for i in range(x_batch.shape[0]):
        x_batch[i, :, :, 0] = noise(x_batch[i, :, :, 0])
        x_batch[i, :, :, 1] = noise(x_batch[i, :, :, 1])

    return x_batch


def get_degreee_seq(n_rotate, max_degree, degree_sampling="uniform"):
    '''
    generate a list of rotation degree (float).

    Args:
        n_rotate: numbers of rotated images to generate.
        max_degree: set the range of uniform sampling, would be (-max_degree, max_degree).
        degree_sampling: String, default is "uniform": do uniform sampling, 
                                            "gaussian": do normal sampling which is dense between -5 to 5 degree and sparse in large degree.
    return:
        degree_seq: a list of rotation degree (float).
    '''
    if degree_sampling == "uniform":
        degree_seq = np.random.uniform(low=-max_degree, high=max_degree, size = n_rotate)
    else:
        degree_seq = np.random.normal(0, 20/3, n_rotate)
        degree_seq *= np.random.normal(0, 0.5, n_rotate)

    return degree_seq


def get_rotate(image, degree_seq, h, w):
    '''
    rotate and resize each image.

    Args:
        image: a template image to roate.
        degree_seq: a list of floats indicate the target rotate degree.
        h, w: target size of the output images.
    output:
        rotate_batch: size (n_rotate, h, w, 2)
    '''
    rotate_batch = np.zeros((len(degree_seq), h, w, 2))
    for i, degree in enumerate(degree_seq):
        rotate_batch[i, :, :, 1] = rotate_image(image, degree, h, w)
        rotate_batch[i, :, :, 0] = resize_img(image, h, w)

    return rotate_batch


def resize_img(img, h, w):
    '''
    resize img to size (w, h)
    '''
    output = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

    return output


def rotate_image(x, degree, h, w, resolution=0.02):
    '''
    Shift the template to generate rotated image, and resize each rotated image to (h, w)

    input:
        x -- a template in polar coordinate system.
        degree -- expected rotation degree, must be divisible by unit.
        resolution -- resolution of the template, min degree (unit) that can be manipulated.

    output:
        rotated_img -- a "rotated" image from template (also in polar coordinate system)
    '''
    rows, cols = x.shape
    shift_matrix = np.float32([[1, 0, degree/resolution], [0, 1, 0]])
    rotated_img = cv2.warpAffine(x, shift_matrix, (cols, rows))

    return resize_img(rotated_img, h, w)
