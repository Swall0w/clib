from PIL import Image
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage import color



def crop():
    pass

def _elastic_transform_2d(img, sigma=6, alpha=36, random=False):

    assert img.ndim == 2

    if random is False:
        random = np.random.RandomState(None)

    shape = img.shape
    dx = gaussian_filter((random.rand(*shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha
    dy = gaussian_filter((random.rand(*shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    return map_coordinates(img, indices, order=1).reshape(shape)


def elastic_transform(img, sigma=6, alpha=36, random=False):

    if img.ndim == 2:
        ret = _elastic_transform_2d(img, sigma, alpha, random)
    elif img.ndim ==3:
        gray_img = color.rgb2gray(img)
        ret = _elastic_transform_2d(gray_img, sigma, alpha, random)
        ret = color.gray2rgb(ret)
    else:
        pass

    return ret
