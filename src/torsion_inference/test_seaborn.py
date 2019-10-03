import sys
from utils.load_stn import load_stn
from plot_result import visual_result
from utils.dataset import load_data
from utils.noise_function import noise
import time
import logging
import os
import numpy as np
import skimage.io as ski
from scipy import misc
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def predict(img_dir, model, batch_size=2, resolution=0.02, add_noise=False, h=60, w=360, 
            max_degree=5, polar_resolution= 0.5, method = 'stn', visual = False):
    '''
    infer rotation angle by cross correlation.
    Args:
        img_dir: a directory that contains all images.
    Output:
        y_pred: list of rotation angle inferred by cross correlation.
        label: ground truth rotaion angle.
    '''
    # load data in batches
    img_dirs = []
    for name in os.listdir(img_dir):
        img_dirs.append(img_dir+name)

    '''
    paths = []
    for img_dir in img_dirs:
        for name in os.listdir(img_dir):
            if name != "template.png":
                paths.append(os.path.join(img_dir, name))
    '''
    paths = img_dirs
    N = len(paths)
    n_batches = N // batch_size
    final_batch = N % batch_size

    n_rotate = int(max_degree*2/resolution)
    y_pred = np.zeros((N, n_rotate))
    label = np.zeros((N, n_rotate))

    Hz_list = []
    for i in range(n_batches):
        batch_paths = paths[i * batch_size:(i + 1) * batch_size]
        # read a batch of template from path
        batch_template = read_from_path(batch_paths)
        X_batch, label_batch = gen_rotate(
            batch_template, resolution=resolution, max_degree=max_degree, h=h, w=w)
        #X_batch, label_batch = load_img(batch_paths, h, w)
        if add_noise:
            try:
                X_batch = noise_batch(X_batch)
            except:
                print(
                    "face some problem to add noise for X_batch.shape = ", X_batch.shape)

        
        # time cross correlation
        start = time.time()
        y_pred[i * batch_size:(i + 1) * batch_size] = predict_batch(X_batch, model, 
                                            label = label_batch, method=method, visual=visual, 
                                            resolution= resolution, polar_resolution= polar_resolution)
        spend_time = time.time() - start
        Hz = batch_size*n_rotate / spend_time
        Hz_list.append(Hz)
        #print("Hz:", Hz)

        label[i * batch_size:(i + 1) * batch_size] = label_batch
    if (final_batch != 0):
        batch_paths = paths[n_batches * batch_size:N]
        batch_template = read_from_path(batch_paths)
        X_batch, label_batch = gen_rotate(
            batch_template, resolution=resolution, max_degree=max_degree, h=h, w=w)

        if add_noise:
            try:
                X_batch = noise_batch(X_batch)
            except:
                print(
                    "face some problem in final batch for X_batch.shape = ", X_batch.shape)
        y_pred[n_batches * batch_size:N]=predict_batch(X_batch, model, 
                                            label = label_batch, method=method, visual=visual, 
                                            resolution= resolution, polar_resolution= polar_resolution)
        label[n_batches * batch_size:N]=label_batch
    y_pred = y_pred.flatten()
    label = label.flatten()
    return y_pred, label, np.array(Hz_list).mean()


def noise_batch(X):
    '''
    Add noise to the images.
    Args:
        X: size (batch_size, n_rotate, h, w, 2). channel 0--template; channel 1--rotate.
    Outpur:
        X: same size as input.
    '''
    output = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            output[i, j, :, :, 0] = noise(X[i, j, :, :, 0])
            output[i, j, :, :, 1] = noise(X[i, j, :, :, 1])

    return output


def read_from_path(dir):
    '''
    Args:
        dir: a list of directory under folder "training/polar". each directory contains one template image and n rotated images.
    Return:
        batch_template: a list of template image with lenghth = batch_size.
    '''
    batch_template = []

    for img_dir in dir:
        for name in os.listdir(img_dir):
            if name == "template.png":
                template = np.array(ski.imread(
                        os.path.join(img_dir, name)) / 255)
                batch_template.append(template)

    return batch_template


def gen_rotate(template_batch, resolution=0.02, max_degree=5, h=60, w=720):
    '''
    Generate rotate image from a batch of templates.

    Args:
        template_batch: size(batch_size, h, w)
        max_degree: scalar. degree range for rotation
    return:
        batch_x: size(batch_size, n_rotate, h, w, 2)
        batch_y: size (batchsize, n_rotate)
    '''
    batch_size = len(template_batch)
    degree_seq = np.arange(-max_degree, max_degree, resolution)
    #degree_seq = np.arange(0, max_degree*2, resolution)
    n_rotate = len(degree_seq)

    # Initialize numpy array
    batch_x = np.zeros((batch_size, n_rotate, h, w, 2))
    batch_y = np.zeros((batch_size, n_rotate))

    for i, image in enumerate(template_batch):
        rotate_batch = get_rotate(image, degree_seq, h, w, resolution)
        batch_x[i] = rotate_batch
        batch_y[i] = degree_seq

    #batch_x = batch_x.reshape((batch_size * n_rotate, h, w, 2))

    return batch_x, batch_y


def get_rotate(image, degree_seq, h, w, resolution):
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
        rotate_batch[i, :, :, 1] = rotate_image(image, degree, h, w, resolution)
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
    shift_matrix = np.float32([[1., 0., degree/resolution], [0., 1., 0.]])
    rotated_img = cv2.warpAffine(x, shift_matrix, (cols, rows))

    return resize_img(rotated_img, h, w)


def load_img(batch_paths, h, w):
    '''
    load one batch of images and read rotate degree from filename.
    And resize the images.
    Args:
        batch_paths: a list of image path of rotated image.
        h, w: target size of the output image.
    Output:
        X: numpy array with size (2, batch_size, h, w). X[0] is the template image, and X[1] is rotated image.
        Y: numpy array with size (batch_size)
    '''
    N = len(batch_paths)
    #X = np.zeros((N, h, w, 2))
    rotate_batch = []
    template_batch = []
    Y = np.zeros((N,))

    for idx, rotate_path in enumerate(batch_paths):
        path = rotate_path.split('/')
        # get rotate degree from filename.
        name = path[-1]
        pure_name = os.path.splitext(name)[0]
        Y[idx] = float(pure_name.split('_')[1])
        # get template path from same directory
        path[-1] = "template.png"
        template_path = "/".join(path)

        # load image and template
        rotate = np.array(ski.imread(rotate_path) / 255)
        rotate = resize_img(rotate, h, w)
        template = np.array(ski.imread(template_path) / 255)
        template = resize_img(template, h, w)
        rotate_batch.append(rotate)
        template_batch.append(template)
        if (path[2] == "polar"):
            path[2] = "polar_" + str(w)
            new_dir = "/".join(path[0:3])
            if not os.path.isdir(new_dir):
                os.mkdir(new_dir)
            new_dir = os.path.join(new_dir, path[3])
            if not os.path.isdir(new_dir):
                os.mkdir(new_dir)
            misc.imsave(os.path.join(new_dir, name), rotate)
            misc.imsave("/".join(path), template)

    X = np.array([template_batch, rotate_batch])

    return X, Y


def predict_batch(X, model, label=None, method='stn', visual=False, resolution=0.02, polar_resolution= 0.5):
    '''
    predict one batch of data.
    Args:
        X: with shape (batch_size, n_rotate, h, w, 2)
        model:
        label: with shape (batch_size, n_rotate), in degree unit.
    Output:
        y_batch: with shape (batch_size, n_rotate). --Rotation angle by cross correlation.
    '''
    batch_size = X.shape[0]
    pred_theta = np.zeros((batch_size, X.shape[1]))
    
    rotate = X[:,:,:,:,1].reshape((batch_size, X.shape[1], X.shape[2], X.shape[3], 1))

    for i in range(batch_size):
        if method == 'stn':
            theta, pred = model.predict({'X': X[i], 'rotate': rotate[i]})
        elif method == 'simple':
            theta = model.predict(X[i])
        elif method == 'cc':
            theta = predict_cc(X[i], resolution, polar_resolution)
        theta /= (X.shape[3]/360) # theta is in pixel unit. adjust it to degree unit.
        if i==0 and visual and method == 'stn':
            visualize_result(pred[400:405], X[i, 400:405],
                             label[i, 400:405], theta[400:405])
        pred_theta[i] = theta.flatten()

    return pred_theta

def predict_cc(X_batch, resolution=0.02, polar_resolution= 0.5):
    '''
    Args:
        X_batch: with shape (n_rotate, h, w, 2)
    '''
    template = X_batch[:,:,:,0]
    rotated = X_batch[:,:,:,1]
    y_batch = []
    for i in range(rotated.shape[0]):
        y_batch.append(predict_rotation_cc(rotated[i], template[i], resolution, polar_resolution))
    y_batch = np.array(y_batch)
    return y_batch


# core method to infer rotation angle: cross corelation
def predict_rotation_cc(rotated_img, template, resolution=0.02, polar_resolution= 0.5):
    cols = rotated_img.shape[1]
    col_pad = cols//2

    coor_value = findTorsionByCV2(template, rotated_img, col_pad)
    width = len(coor_value)

    x = np.arange(0, width, 1)
    f = interp1d(x, coor_value, kind='cubic')
    x_new = np.arange(0, width-1, (resolution*polar_resolution))
    y_new = f(x_new)
    max_index = np.argmax(y_new)
    rotate_degree = max_index*(resolution*polar_resolution)-col_pad
    return rotate_degree

def findTorsionByCV2(template, img_r, col_pad):
    '''
    inputs:
        col_pad: numbers of columns to pad to the left and the right, usually = n_columns/2
    '''

    # pad the columns only to return 1d correlate
    img_r = np.pad(img_r, ((0, 0),(col_pad, col_pad)),'constant', constant_values=(0,0))

    img_r = img_r.astype(np.float32)
    template = template.astype(np.float32)

    # match template
    coor = cv2.matchTemplate(img_r, template, cv2.TM_CCORR_NORMED)

    return coor.squeeze()

def resize_img(img, h, w):
    '''
    resize img to size (w, h)
    '''
    output = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

    return output


def visualize_result(output_imgs, input_imgs, label, theta):
    '''
    visualize the output(resample) image by STN.
    Args:
        output_imgs: with shape (n, h, w, 1)
        input_imgs: with shape (n, h, w, 2)
        label: rotation angle(groud_truth), (n, )
        theta: rotation angle(prediction), (n, )
    '''
    n = len(label)
    print(label)
    print(theta)
    print(output_imgs.shape)
    print(input_imgs.shape)
    __, axs = plt.subplots(n, 3, figsize=(n*4, 8))
    for i in range(n):
        # plot input_imgs[i, :, :, 0] "template"
        axs[i, 0].imshow(input_imgs[i, :, :, 0], cmap="gray")
        axs[i, 0].set_title("template")
        # plot input_imgs[i, :, :, 1] "rotate"
        axs[i, 1].imshow(input_imgs[i, :, :, 1], cmap="gray")
        axs[i, 1].set_title("input: " + str(label[i]))
        # plot output_imgs[i]
        axs[i, 2].imshow(output_imgs[i, :, :, 0], cmap="gray")
        axs[i, 2].set_title("output: " + str(theta[i]))
    plt.tight_layout()
    plt.savefig("results/STN_out/STN8_"+str(theta[0])[1:5]+".png")
    print("visualize result saved!")
        




if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[1])

    logging.getLogger().setLevel(logging.DEBUG)

    result_path = 'results/CNN_prediction/'
    h = 60
    w = 720  # the training width
    max_degree = 5
    test_dir = "data/test/polar/"
    noise_mode = "STN8_glints0.3"
    #condition = "uniform_"+noise_mode+"_smaller_"+str(w)+"_ro16_epoch100_max"+str(max_degree)
    condition = "uniform_STN8_glints0.3_all_720_ro16_bs16_max5"
    # w = 360   # the testing width

    if noise_mode == "noNoise":
        add_noise = False
    else:
        add_noise = True

    #add_noise = False


    logging.basicConfig(level=logging.INFO,
                        filename='test_log/'+condition+'.log',
                        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                        )

    logging.info("loading model..")

    model_path = "model/"+condition+".h5"
    model = load_stn(model_path)

    logging.info("testing model...")
    y_pred, Y_val, mean_Hz = predict(
        test_dir, model, batch_size=2, resolution=0.02, add_noise=add_noise, h=h, w=w, max_degree=max_degree)
    logging.info("mean Hz:")
    logging.info(mean_Hz)

    visual_result(Y_val, y_pred, result_path, condition+"_no_glint")

    logging.info("finish")
