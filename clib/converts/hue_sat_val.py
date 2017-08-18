import numpy as np

import cv2


# 画像を読み込んで、hue, sat, val空間でランダム変換を加える関数


def random_hsv_image(bgr_image, delta_hue, delta_sat_scale, delta_val_scale):
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV).astype(np.float32)

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
    bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return bgr_image
