import numpy as np
from skimage import transform
from scipy.misc import imresize
import cv2

def noise(x, sigma=0.05, shear=0.1, blur_kernel = 2, glints_index = 0.3):
    '''
    data augmentation (replace keras)
    input:
        x: image array, size (h, w, 1)
    output:
        x: image with noise, size (h, w, 1)
    '''
    x = random_glints(x, weight=glints_index)
    x = shearing(x, shear = shear)
    x = gaussian_noise(x, sigma=sigma)
    x = blur(x, kernel_size=(blur_kernel, blur_kernel))
    return x


def gaussian_noise(x, mean=0, sigma=0.05):
    '''
    a function to add gaussian noise to the image.
    '''
    row, col = x.shape
    gaussian = np.random.normal(mean, sigma, (row, col))
    gaussian = gaussian.reshape(row, col)

    return x+gaussian


def random_glints(x, weight = 0.3):
    h, w = x.shape
    x1 = np.random.normal(2, 1, h//12*w//12).reshape(h//12, w//12)
    x2 = np.zeros(x1.shape)
    x2[x1 > 0] = 1
    x3 = imresize(x2, (h, w), interp='cubic')
    x4 = np.zeros(x3.shape)
    x4[x3 > 255/2] = 1

    return x+x4*weight



def shearing(x, shear = 0.1):
    afine_tf = transform.AffineTransform(shear=shear)
    return transform.warp(x, inverse_map=afine_tf)

def blur(x, kernel_size = (2, 2)):
    return cv2.blur(x, kernel_size)




if __name__ == "__main__":
    import os
    import skimage.io as ski
    import matplotlib.pyplot as plt

    img_dirs = "/mnt/data/training/polar/"

    for i, sub_path in enumerate(os.listdir(img_dirs)):
        if i>1 :
            break
        for j, name in enumerate(os.listdir(img_dirs+sub_path)):
            if name == "template.png":
                img = np.array(ski.imread(os.path.join(img_dirs, sub_path, name))/255)
            elif j==0:
                img2 = np.array(ski.imread(os.path.join(img_dirs, sub_path, name))/255)
                rotate_name = name


    h = 60
    w = 360
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_CUBIC)

    __, axs = plt.subplots(6,4, figsize=(32,9))
    for i in range(3):
        axs[i*2, 0].imshow(img, cmap = "gray")
        axs[i*2, 0].set_title("origin")
        axs[i*2+1, 0].imshow(img2, cmap = "gray")
        axs[i*2+1, 0].set_title(rotate_name)

        glints = random_glints(img)
        shear = shearing(glints, shear = 0.1)
        axs[i*2, 3].imshow(shear, cmap = "gray")
        axs[i*2, 3].set_title("shearing "+str(0.1))

        glints2 = random_glints(img2, weight=0.5)
        shear2 = shearing(glints2, shear = 0.1)
        axs[i*2+1, 3].imshow(shear2, cmap = "gray")
        axs[i*2+1, 3].set_title("shearing "+str(0.1))

        gaussian = gaussian_noise(shear, sigma=0.01*(i+3))
        axs[i*2, 2].imshow(gaussian, cmap = "gray")
        axs[i*2, 2].set_title("gaussian "+str(0.01*(i+3)))

        gaussian2 = gaussian_noise(shear2, sigma=0.01*(i+3))
        axs[i*2+1, 2].imshow(gaussian2, cmap = "gray")
        axs[i*2+1, 2].set_title("gaussian "+str(0.01*(i+3)))

        blurring = blur(gaussian, kernel_size = (2, 2))
        axs[i*2, 1].imshow(blurring, cmap = "gray")
        axs[i*2, 1].set_title("gaussian"+str(0.01*(i+3))+"+blur (" + str(2) + "," + str(2)+")")

        blurring2 = blur(gaussian2, kernel_size = (2, 2))
        axs[i*2+1, 1].imshow(blurring2, cmap = "gray")
        axs[i*2+1, 1].set_title("gaussian"+str(0.01*(i+3)) +
                                "+blur (" + str(2) + "," + str(2)+")")

    plt.tight_layout()
    plt.savefig("/mnt/results/visual_augment/train/random_glints.png")

