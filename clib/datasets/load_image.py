import numpy
from clib.transforms import (add_noise, add_salt_and_pepper_noise, brightness,
                             contrast, gamma_adjust, gaussian_blur, saturation,
                             sharpness)
from clib.utils import randombool
from skimage import io, transform


class ImageAugmentation():
    def __init__(self):
        self.imread = io.imread

    def read(self, imgpath):
        return self.imread(imgpath)

    def blur(self, img, israndom=False):
        if not israndom:
            img = gaussian_blur(img)
        return img

    def noise(self, img, israndom=False):
        if not israndom:
            img = add_noise(img)
        return img

    def sp_noise(self, img, israndom=False):
        if not israndom:
            img = add_salt_and_pepper_noise(img)
        return img

    def contrast(self, img, israndom=False):
        if not israndom:
            img = contrast(img)
        return img

    def brightness(self, img, israndom=False):
        if not israndom:
            img = brightness(img)
        return img

    def saturation(self, img, israndom=False):
        if not israndom:
            img = saturation(img)
        return img

    def sharpness(self, img, israndom=False):
        if not israndom:
            img = sharpness(img)
        return img

    def gamma_adjust(self, img, israndom=False):
        if not israndom:
            img = gamma_adjust(img)
        return img

    def resize(self, img, size):
        img = transform.resize(img, size, mode='reflect')
        return img

    def crop(self, img, position):
        if img.ndim == 3:
            img = img[position[1]:position[3], position[0]:position[2], :]
        else:
            img = img[position[1]:position[3], position[0]:position[2]]
        return img
