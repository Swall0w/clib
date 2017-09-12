import os
import random

import numpy
import six
from chainer.dataset import dataset_mixin
from clib.datasets import crop_image_random_transform, uniform, voc_load
from clib.datasets import ImageAugmentation
from clib.utils.regrex import is_path
from clib.transforms import jitter_position
from clib.utils import randombool

try:
    from PIL import Image
    available = True
except ImportError as e:
    available = False
    _import_error = e


class XMLLabeledImageDataset(dataset_mixin.DatasetMixin):

    def __init__(self, pairs, label_dict, dtype=numpy.float32,
                 label_dtype=numpy.int32, resize=None, random_step=0,
                 is_image_aug=False):
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
        self.random_step = random_step
        self.label_dict = label_dict
        self.is_image_aug = is_image_aug

    def __len__(self):
        return len(self._pairs)

    def get_example(self, i):
        full_path, label = self._pairs[i]
        img_size, label = voc_load(label)
        bndbox = random.choice(label)

        bbox = (bndbox['xmin'], bndbox['ymin'], bndbox['xmax'], bndbox['ymax'])
        random_bbox = jitter_position(bbox, img_size,
                                      step=(self.random_step, self.random_step))
        imag = ImageAugmentation()
        img = imag.read(full_path)
        img = imag.crop(img, random_bbox)
        img = imag.resize(img, self.resize)

        if self.is_image_aug:
            img = imag.blur(img, israndom=randombool())
            img = imag.noise(img, israndom=randombool())
            img = imag.sp_noise(img, israndom=randombool())
            img = imag.contrast(img, israndom=randombool())
            img = imag.brightness(img, israndom=randombool())
            img = imag.saturation(img, israndom=randombool())
            img = imag.sharpness(img, israndom=randombool())
            img = imag.gamma_adjust(img, israndom=randombool())


#        if self.random_step > 0:
#            x_step = random.randint(-self.random_step, self.random_step)
#            y_step = random.randint(-self.random_step, self.random_step)
#        else:
#            x_step = 0
#            y_step = 0
#        bbox = (bndbox['xmin'], bndbox['ymin'], bndbox['xmax'], bndbox['ymax'])
#        step = (x_step, y_step)
#
#        image = crop_image_random_transform(path=full_path, bbox=bbox,
#                                            step=step, dtype=self._dtype,
#                                            blur=True, contrast=True,
#                                            gamma=True, gauss_noise=True,
#                                            sp_noise=True, sharpness=True,
#                                            saturation=True)
#        image = uniform(image, self.resize, self._dtype)

        label_dict = self.label_dict[bndbox['label']]
        label = numpy.array(label_dict, dtype=self._label_dtype)
        return img.transpose(2, 0, 1), label


def _check_pillow_availability():
    if not available:
        raise ImportError('PIL cannot be loaded. Install Pillow!\n'
                          'The actual import error is as follows:\n' +
                          str(_import_error))
