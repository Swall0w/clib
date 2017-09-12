import numpy as np
from skimage.transform import resize as imresize


def batch(batch):
    format_size = batch[0][0].shape[1]
    format_batch = []

    for index, item in enumerate(batch):
        original_image = item[0]
        resized_image = imresize(original_image,
                                 (format_size, format_size),
                                 mode='reflect')
        format_batch.append((resized_image, batch[index][1]))
    return format_batch


def resize_to_yolo(img):
    if img.ndim == 2:
        raise ValueError(
            "image shoule be RGB format. But image is {}".format(img.ndim))
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
    img = imresize(img, (input_height, input_width), mode='reflect')

    return img
