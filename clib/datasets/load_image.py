from PIL import Image, ImageEnhance, ImageFilter
import numpy
from clib.utils import randombool

def gamma_table(gamma_r, gamma_g, gamma_b, gain_r=1.0, gain_b=1.0):
    r_tbl = [min(255, int((x / 255.) ** (1. / gamma_r) * gamma_r * 255.)) for x in range(256)]
    g_tbl = [min(255, int((x / 255.) ** (1. / gamma_g) * gamma_g * 255.)) for x in range(256)]
    b_tbl = [min(255, int((x / 255.) ** (1. / gamma_b) * gamma_b * 255.)) for x in range(256)]
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
