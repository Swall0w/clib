from PIL import Image, ImageEnhance
import numpy as np
import skimage
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage import color
from skimage.util import random_noise


def _elastic_transform_2d(img, sigma=6, alpha=36, random=False):

    def _calc_delta(shape, alpha, sigma):
        return alpha * gaussian_filter((random.rand(*shape) * 2 - 1),
                                       sigma, mode='constant', cval=0)

    assert img.ndim == 2

    if random is False:
        random = np.random.RandomState(None)

    shape = img.shape
    dx = _calc_delta(shape, alpha, sigma)
    dy = _calc_delta(shape, alpha, sigma)

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    return map_coordinates(img, indices, order=1).reshape(shape)


def elastic_transform(img, sigma=6, alpha=36, random=False):

    if img.ndim == 2:
        ret = _elastic_transform_2d(img, sigma, alpha, random)
    elif img.ndim == 3:
        gray_img = color.rgb2gray(img)
        ret = _elastic_transform_2d(gray_img, sigma, alpha, random)
        ret = color.gray2rgb(ret)
    else:
        pass

    return ret

def gaussian_blur(img, sigma=1, multichannel=True):
    return skimage.filters.gaussian(image=img, sigma=sigma,
                                    multichannel=multichannel)


def add_noise(img, sigma=0.155):
    return random_noise(img, var=sigma**2)


def add_salt_and_pepper_noise(img, salt_vs_pepper=0.5):
    return random_noise(img, mode='s&p', salt_vs_pepper=salt_vs_pepper)

def contrast(img, value=1.0):
    img = ImageEnhance.Contrast(Image.fromarray(np.uint8(img))).enhance(value)
    return np.asarray(img)
