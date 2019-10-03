import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.ndimage
import skimage.io as ski

#%%
def cart2polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def polar2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def polarTransform(img, data, origin=None, resolution = 0.02):
    '''
    Reprojects a 3D numpy array ("data") into a polar coordinate system.
    
    Args: 
        img -- origin image
        data -- a 3D numpy array (x, y)
        resolution -- resolution of the template (degree per pixel). default is 1 degree.
        origin -- a tuple of (x0, y0) and defaults to the center of the image.

    Returns:
        output -- transformed image in polar coordinate system
        r_i
        theta_i
    '''
    x, y = data
    
    if origin is None:
        origin = (nx//2, ny//2)
    x = x - origin[0]
    y = y - origin[1]
    
    r, theta = cart2polar(x, y)

    # Make a regular (in polar space) grid based on the min and max r & theta
    ny = int(np.round(r.max()-r.min()))
    nx = int(np.round(360*(1/resolution))) 
    
    r_i = np.linspace(r.min(), r.max(), ny)
    theta_i = np.linspace(0, np.pi*2, nx) 
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Project the r and theta grid back into pixel coordinates
    xi, yi = polar2cart(r_grid, theta_grid)
    xi += origin[0] # We need to shift the origin back to 
    yi += origin[1] # back to the lower-left corner...
    xi, yi = xi.flatten(), yi.flatten()
    coords = np.vstack((xi, yi)) # (map_coordinates requires a 2xn array)
    transformed = sp.ndimage.map_coordinates(img, coords[::-1], order=1)
    output = transformed.reshape(ny,nx)
    return output, r_i, theta_i


