import numpy
from clib.utils import randombool
from clib.transforms import (gaussian_blur, add_noise,
                             add_salt_and_pepper_noise, contrast,
                             brightness, saturation, sharpness,
                             gamma_adjust)
from skimage import io, transform
from PIL import Image, ImageEnhance, ImageFilter


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
#        img = transform.resize(img, size, mode='reflect')
        image = Image.fromarray(numpy.uint8(img)).resize((size[1], size[0]))
        img = numpy.asarray(image, dtype=numpy.float32)
        return img

    def crop(self, img, position):
        if img.ndim == 3:
            img = img[position[0]:position[2], position[1]:position[2], :]
        else:
            img = img[position[0]:position[2], position[1]:position[2]]
        return img

def gamma_table(gamma_r, gamma_g, gamma_b, gain_r=1.0, gain_b=1.0):
    def _gen_table(gamma):
        def _calc_min(x, gamma):
            return min(255, int((x / 255.) ** (1. / gamma) * gamma * 255.))

        table = [_calc_min(x, gamma) for x in range(256)]
        return table
    r_tbl = _gen_table(gamma_r)
    g_tbl = _gen_table(gamma_g)
    b_tbl = _gen_table(gamma_b)
    return r_tbl + g_tbl + b_tbl


def crop_image_random_transform(path, bbox, dtype, step=(0, 0),
                                brightness=False, blur=False, contrast=False,
                                gamma=False, gauss_noise=False, sp_noise=False,
                                sharpness=False, saturation=False):
    image = Image.open(path)
    image = trans_crop(image, step, bbox)
    if len(numpy.asarray(image).shape) == 2:
        image = to_rgb(image, dtype)

    if blur:
        if randombool():
            image = image.filter(ImageFilter.BLUR)

    if contrast:
        if randombool():
            if randombool():
                image = ImageEnhance.Contrast(image).enhance(1.5)
            else:
                image = ImageEnhance.Contrast(image).enhance(0.5)

    if gamma:
        image = image.point(gamma_table(1.2, 0.5, 0.5))

    if brightness:
        if randombool():
            if randombool():
                image = ImageEnhance.Brightness(image).enhance(1.5)
            else:
                image = ImageEnhance.Brightness(image).enhance(0.5)

    if saturation:
        if randombool():
            if randombool():
                image = ImageEnhance.Color(image).enhance(1.5)
            else:
                image = ImageEnhance.Color(image).enhance(0.5)

    if sharpness:
        if randombool():
            if randombool():
                image = ImageEnhance.Sharpness(image).enhance(1.5)
            else:
                image = ImageEnhance.Sharpness(image).enhance(0.5)

    if gauss_noise:
        pass

    if sp_noise:
        pass

    return image


def trans_crop(image, step, bbox):
    left = bbox[0] - step[0]
    top = bbox[1] - step[1]
    right = bbox[2] - step[0]
    bottom = bbox[3] - step[1]
    image = image.crop((left, top, right, bottom))
    return image


def uniform(image, resize, dtype):
    image = image.resize((int(resize[0]), int(resize[1])), Image.LANCZOS)
    image = convert_2_array(image, dtype)
    return image


def convert_2_array(img, dtype):
    return numpy.asarray(img, dtype=dtype)


def to_rgb(image, dtype):
    image = convert_2_array(image, dtype)
    w, h = image.shape
    ret = numpy.empty((w, h, 3), dtype)
    ret[:, :, 0] = image
    ret[:, :, 1] = image
    ret[:, :, 2] = image
    ret = Image.fromarray(numpy.uint8(ret))
    return ret
