import os
import random

import numpy
import six
from chainer.dataset import dataset_mixin
from clib.datasets import ImageAugmentation, voc_load
from clib.transforms import jitter_position
from clib.utils import randombool
from clib.utils.regrex import is_path
from skimage.color import gray2rgb
from .basedataset import BaseLabeledImageDataset


class XMLLabeledImageDataset(BaseLabeledImageDataset):
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

        if img.ndim == 2:
            img = gray2rgb(img)
        img = img.astype(self._dtype)
        label_dict = self.label_dict[bndbox['label']]
        label = numpy.array(label_dict, dtype=self._label_dtype)
        return img.transpose(2, 0, 1), label
