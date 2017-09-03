from clib.utils import load_class
from chainer import serializers, cuda
from PIL import Image
import numpy as np


class ImageInference(object):
    def __init__(self):
        self.weights = weights
        self.labelfile = labelfile
        self.gpu = gpu
        self.model = model

        self.label = load_class(self.labelfile)
        serializers.load_npz(self.weights, self.model)

        if self.gpu >= 0:
            cuda.get_device(self.gpu).use()
            self.model.to_gpu()

    def load_image(self, imagefile):
        img = Image.open(imagefile)
        cv_img = np.array(imagefile)[:, :, ::-1].copy()
        return cv_img

    def __call__(self, inputs):
        """Applies inference computation to input arrays.
    
        Args:
            inputs: input arrays.
        Return:
            dictionary: Dictionary of :class: objects.
        """
        raise NotImplementedError()
