import numpy as np
import cv2
from cv2 import GaussianBlur
from skimage.exposure import equalize_adapthist as adhist
from scipy.interpolate import interp1d
from .PolarTransform import polarTransform
from scipy.signal import correlate
from skimage import img_as_float

def findTorsion(template, rotated_img, useful_map_r, center, filter_sigma = 1, adhist_times = 2, resolution = 0.02, polar_resolution=0.5, method = "hoi", model = None):
    if method == "hoi":
        return findTorsion_hoi(template, rotated_img, useful_map_r, center, filter_sigma, adhist_times, resolution)
    elif method == "cc":
        return findTorsion_cc(template, rotated_img, useful_map_r, center, filter_sigma, adhist_times, resolution, polar_resolution)
    elif method == "network":
        return findTorsion_model(template, rotated_img, model, useful_map_r, center, filter_sigma, adhist_times, resolution, polar_resolution)
    elif method == "stn":
        return findTorsion_stn(template, rotated_img, model, useful_map_r, center, filter_sigma, adhist_times, resolution, polar_resolution)


def guassianMap(img_mean, img_std, useful_map, filter_sigma = 1):
    guassian_map = np.random.normal(img_mean, img_std, useful_map.shape)
    guassian_map[guassian_map < 0] = 0
    guassian_map[guassian_map > 1] = 1
    guassian_map = GaussianBlur(guassian_map, ksize= (0,0), sigmaX = filter_sigma)
    guassian_map[useful_map==1] = 0
    return guassian_map

def genPolar(img, useful_map, center, template=False, filter_sigma = 1, adhist_times = 2, resolution = 0.02):
    if adhist_times >= 1:
        img_enhanced = img_as_float(adhist(img))
    else:
        img_enhanced = img_as_float(img)
    # print(img_enhanced)
    img_enhanced[useful_map == 0] = 0
    # guassian_map = guassianMap(img.mean(), img.std(), useful_map, filter_sigma = filter_sigma)
    guassian_map = guassianMap(0.5, 0.2, useful_map, filter_sigma = filter_sigma)
    # If no radial/tangential filtering is performed, alter the codes to contain only one polarTransform function to speed up performance
    output_img, r, theta = polarTransform(img_enhanced, np.where(useful_map==1)[::-1], origin=center, resolution=resolution)
    if adhist_times >= 2:
        kernel_size = None
        if (output_img.shape[0] < 8): # solving the "Division by zero" error in adhist function (kernel_size = 0 if img.shape[?] < 8)
            kernel_size = [1,1]
            if (output_img.shape[1] > 8):
                kernel_size[1] = output_img.shape[1]//8
        output_img = adhist(output_img, kernel_size)
    output_gaussian, r_gaussian, theta_gaussian = polarTransform(guassian_map, np.where(useful_map==1)[::-1], origin=center, resolution=resolution)
    output = output_img + output_gaussian

    if template == True:
        output_longer = output
        extra_rad = 0
        #extra_index, extra_rad = int(25/resolution), np.deg2rad(25)
        #output_longer = np.concatenate((output[:,output.shape[1]-extra_index:], output, output[:, 0:extra_index]), axis = 1)
        return output, output_longer, r, theta, extra_rad
    else:
        return output, r, theta




def findTorsion_hoi(template, rotated_img, useful_map_r, center, filter_sigma = 1, adhist_times = 2, resolution = 0.02):
    output_r, r_r, theta_r = genPolar(rotated_img, useful_map_r, center , filter_sigma = filter_sigma, adhist_times = adhist_times, resolution = resolution)

    cols = output_r.shape[1]
    col_pad = cols//2

    template = cv2.resize(template, (cols, 60), interpolation=cv2.INTER_CUBIC)
    output_r = cv2.resize(output_r, (cols, 60), interpolation=cv2.INTER_CUBIC)

    coor_value = findTorsionByCV2(template, output_r, col_pad)

    max_index = np.argmax(coor_value)
    rotate_degree = max_index*resolution-col_pad
    return rotate_degree, (output_r, r_r, theta_r), coor_value


def findTorsion_cc(template, rotated_img, useful_map_r, center, filter_sigma = 1, adhist_times = 2, resolution = 0.02, polar_resolution=1):
    '''
    Args:
        template -- template polar image
        rotated_img -- other image without polar transform
        useful_map_r, center, filter_sigma , adhist_times: parameters for polar transform.
        resolution -- resolution of the output(after interpolated).
        polar_resolution -- resolution during polar transform.
    '''
    
    output_r, r_r, theta_r = genPolar(rotated_img, useful_map_r, center , filter_sigma = filter_sigma, adhist_times = adhist_times, resolution = polar_resolution)
    
    cols = output_r.shape[1]
    col_pad = cols//2

    template = cv2.resize(template, (cols, 60), interpolation=cv2.INTER_CUBIC)
    output_r = cv2.resize(output_r, (cols, 60), interpolation=cv2.INTER_CUBIC)

    coor_value = findTorsionByCV2(template, output_r, col_pad)
    width = len(coor_value)

    x = np.arange(0, width, 1)
    f = interp1d(x, coor_value, kind='cubic')
    x_new = np.arange(0, width-1, (resolution/polar_resolution))
    y_new = f(x_new)
    max_index = np.argmax(y_new)
    rotate_degree = max_index*(resolution/polar_resolution)-col_pad
    return rotate_degree*polar_resolution, (output_r, r_r, theta_r), coor_value


def findTorsion_model(template, rotated_img, model, useful_map_r, center, filter_sigma = 1, adhist_times = 2, resolution = 0.02, polar_resolution=0.5):
    '''
    Args:
        template -- template polar image
        rotated_img -- other image without polar transform
        model -- torsion model to infer rotation angle.
        useful_map_r, center, filter_sigma , adhist_times: parameters for polar transform.
        resolution -- resolution of the output(after interpolated).
        polar_resolution -- resolution during polar transform.
    '''
    
    output_r, r_r, theta_r = genPolar(rotated_img, useful_map_r, center , filter_sigma = filter_sigma, adhist_times = adhist_times, resolution = polar_resolution)
    
    h = 60
    w = int(360/polar_resolution)

    output_r = output_r.astype(np.float32)
    template = template.astype(np.float32)

    template = cv2.resize(template, (w, h), interpolation=cv2.INTER_CUBIC)
    output_r = cv2.resize(output_r, (w, h), interpolation=cv2.INTER_CUBIC)

    # now you have "output_r" and "template"
    x = np.zeros((1, h, w, 2))
    x[0, :, :, 0] = template
    x[0, :, :, 1] = output_r
    
    # load deep learning model
    rotate_degree = model.predict(x).squeeze()
    rotate_degree /= (w/360)
    coor_value = None

    return rotate_degree, (output_r, r_r, theta_r), coor_value


def findTorsion_stn(template, rotated_img, model, useful_map_r, center, filter_sigma = 1, adhist_times = 2, resolution = 0.02, polar_resolution=0.5):
    '''
    Args:
        template -- template polar image
        rotated_img -- other image without polar transform
        model -- torsion model to infer rotation angle.
        useful_map_r, center, filter_sigma , adhist_times: parameters for polar transform.
        resolution -- resolution of the output(after interpolated).
        polar_resolution -- resolution during polar transform.
    '''
    
    output_r, r_r, theta_r = genPolar(rotated_img, useful_map_r, center , filter_sigma = filter_sigma, adhist_times = adhist_times, resolution = polar_resolution)
    
    h = 60
    w = 720

    output_r = output_r.astype(np.float32)
    template = template.astype(np.float32)

    template = cv2.resize(template, (w, h), interpolation=cv2.INTER_CUBIC)
    output_r = cv2.resize(output_r, (w, h), interpolation=cv2.INTER_CUBIC)

    # now you have "output_r" and "template"
    x = np.zeros((1, h, w, 2))
    x[0, :, :, 0] = template
    x[0, :, :, 1] = output_r

    # load deep learning model
    rotate_degree, __ = model.predict({'X': x, 'rotate': x[:,:,:,1].reshape(1, h, w, 1)})
    
    rotate_degree = rotate_degree.squeeze()
    rotate_degree /= (w/360)
    coor_value = None

    return rotate_degree, (output_r, r_r, theta_r), coor_value



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