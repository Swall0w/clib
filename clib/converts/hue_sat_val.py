import numpy as np
from skimage import color


# 画像を読み込んで、hue, sat, val空間でランダム変換を加える関数


def random_hsv_image(rgb_image, delta_hue, delta_sat_scale, delta_val_scale):
    hsv_image = color.rgb2hsv(rgb_image).astype(np.float32)

    # hue
    hsv_image[:, :, 0] += int(
        (np.random.rand() * delta_hue * 2 - delta_hue) * 255)

    # sat
    sat_scale = 1 + np.random.rand() * delta_sat_scale * 2 - delta_sat_scale
    hsv_image[:, :, 1] *= sat_scale

    # val
    val_scale = 1 + np.random.rand() * delta_val_scale * 2 - delta_val_scale
    hsv_image[:, :, 2] *= val_scale

    hsv_image[hsv_image < 0] = 0
    hsv_image[hsv_image > 255] = 255
    hsv_image = hsv_image.astype(np.uint8)
    rgb_image = color.hsv2rgb(hsv_image)
    return rgb_image
