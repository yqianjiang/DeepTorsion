'''Utilities for real-time data augmentation on image data.
'''
import os
import threading
import numpy as np
from keras.preprocessing.image import Iterator
import skimage.io as ski
from .noise_function import noise
from .preprocess import preprocess_batch

class MyIterator(Iterator):
    ''' class for image data iterators.

    # Arguments:
        dir: a list of template images path under  "training/polar".
        batch_size: Integer, size of a batch.
        n_rotate: Integer, number of rotated image to generate from one template image.
        h: Interger, target height of the training image.
        w: Interger, target width of the training image.
        degree_sampling: String, "uniform" sampling or "gaussian" sampling to the rotation degree sequence.
        add_noise: Boolean, whether to add noise or not.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    '''

    def __init__(self, dir, batch_size, n_rotate, h, w, degree_sampling, add_noise, shuffle, seed, max_degree, stn = False):
        self.dir = dir
        self.n_rotate = n_rotate
        self.h = h
        self.w = w
        self.degree_sampling = degree_sampling
        self.add_noise = add_noise
        self.max_degree = max_degree
        self.isSTN = stn
        n = len(dir)
        super(MyIterator, self).__init__(n, batch_size, shuffle, seed)


    def _get_batches_of_transformed_samples(self, index_array):
        batch_paths = []
        for i in index_array:
            batch_paths.append(self.dir[i])

        # read a batch of template from path
        batch_template = self.read_from_path(batch_paths)

        # preprocess (generate rotated images and add noise) one batch of the template
        batch_x, batch_y = preprocess_batch(batch_template, n_rotate=self.n_rotate, add_noise = self.add_noise, degree_sampling=self.degree_sampling, max_degree=self.max_degree, h=self.h, w=self.w)

        if self.isSTN:
            # for multiple inputs and outputs
            batch_template = batch_x[:,:,:,0].reshape((batch_x.shape[0],self.h, self.w, 1))
            batch_rotate = batch_x[:, :, :, 1].reshape(
                (batch_x.shape[0], self.h, self.w, 1))
            # batch_x = [batch_x, batch_rotate]
            batch_x2 = {'X': batch_x, 'rotate': batch_rotate}
            # batch_y = [batch_template, batch_y]
            theta = batch_y
            #M = self.gen_matrix(batch_y)
            batch_y2 = {'resample': batch_template, 'theta': theta}
            return (batch_x2, batch_y2)
        else:
            return (batch_x, batch_y)
        
    def gen_matrix(self, batch_y):
        n = len(batch_y)
        theta = np.zeros((n, 6))
        for i in range(n):
            theta[i] = np.float32([1, 0, -batch_y[i]/360, 0, 1, 0])
        return theta


    def read_from_path(self, dir):
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
