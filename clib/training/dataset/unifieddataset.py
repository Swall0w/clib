import os

import numpy
import six
from PIL import Image
from .basedataset import BaseLabeledImageDataset


def to_rgb(image, dtype):
    w, h = image.shape
    ret = numpy.empty((w, h, 3), dtype)
    ret[:, :, 0] = image
    ret[:, :, 1] = image
    ret[:, :, 2] = image
    return ret


def _read_image_as_array(path, dtype, resize):
    f = Image.open(path)
    if resize:
        f = f.resize((int(resize[0]), int(resize[1])))
    try:
        image = numpy.asarray(f, dtype=dtype)
        if len(image.shape) == 2:
            image = to_rgb(image, dtype)
    finally:
        # Only pillow >= 3.0 has 'close' method
        if hasattr(f, 'close'):
            f.close()
    return image


class UnifiedLabeledImageDataset(BaseLabeledImageDataset):
    def get_example(self, i):
        full_path, int_label = self._pairs[i]
        image = _read_image_as_array(full_path, self._dtype, self.resize)

        if image.ndim == 2:
            # image is greyscale
            image = image[:, :, numpy.newaxis]
        label = numpy.array(int_label, dtype=self._label_dtype)
        return image.transpose(2, 0, 1), label
