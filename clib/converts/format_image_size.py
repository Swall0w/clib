import numpy as np

import cv2


def batch(batch):
    format_size = batch[0][0].shape[1]
    format_batch = []

    for index, item in enumerate(batch):
        original_image = item[0]
        transpose_image = np.transpose(original_image, (1, 2, 0))
        resized_image = cv2.resize(transpose_image, (format_size, format_size))
        resized_image = resized_image.transpose(2, 0, 1).astype(np.float32)
        format_batch.append((resized_image, batch[index][1]))
    return format_batch


def resize_to_yolo(img):
    input_height, input_width, _ = img.shape
    min_pixel = 320
    max_pixel = 448

    min_edge = np.minimum(input_width, input_height)
    if min_edge < min_pixel:
        input_width *= min_pixel / min_edge
        input_height *= min_pixel / min_edge
    max_edge = np.maximum(input_width, input_height)
    if max_edge > max_pixel:
        input_width *= max_pixel / max_edge
        input_height *= max_pixel / max_edge

    input_width = int(input_width / 32 + round(input_width % 32 / 32)) * 32
    input_height = int(input_height / 32 + round(input_height % 32 / 32)) * 32
    img = cv2.resize(img, (input_width, input_height))

    return img
