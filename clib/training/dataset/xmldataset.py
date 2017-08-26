import os

import numpy
import six
from chainer.dataset import dataset_mixin
from clib.utils.regrex import is_path

try:
    from PIL import Image
    available = True
except ImportError as e:
    available = False
    _import_error = e



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


class XMLLabeledImageDataset(dataset_mixin.DatasetMixin):

    def __init__(self, pairs, dtype=numpy.float32,
                 label_dtype=numpy.int32, resize=None):
        _check_pillow_availability()
        if isinstance(pairs, six.string_types):
            pairs_path = pairs
            with open(pairs_path) as pairs_file:
                pairs = []
                for i, line in enumerate(pairs_file):
                    pair = line.strip().split()
                    if len(pair) != 2:
                        raise ValueError(
                            'invalid format at line {} in file {}'.format(
                                i, pairs_path))
                    if is_path(pair[1]):
                        label = str(pair[1])
                    else:
                        label = int(pair[1])
                    pairs.append((pair[0], label))
        self._pairs = pairs
        self._dtype = dtype
        self._label_dtype = label_dtype
        self.resize = resize

    def __len__(self):
        return len(self._pairs)

    def get_example(self, i):
        full_path, label = self._pairs[i]
        image = _read_image_as_array(full_path, self._dtype, self.resize)

        if image.ndim == 2:
            # image is greyscale
            image = image[:, :, numpy.newaxis]
        label = numpy.array(int_label, dtype=self._label_dtype)
        return image.transpose(2, 0, 1), label


def _check_pillow_availability():
    if not available:
        raise ImportError('PIL cannot be loaded. Install Pillow!\n'
                          'The actual import error is as follows:\n' +
                          str(_import_error))