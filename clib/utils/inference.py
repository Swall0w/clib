from chainer import cuda, serializers
from clib.utils import load_class
from skimage import io


class ImageInference(object):
    def __init__(self, model, weights, labelfile, gpu=0):
        self.model = model
        self.weights = weights
        self.labelfile = labelfile
        self.gpu = gpu

        self.label = load_class(self.labelfile)
        serializers.load_npz(self.weights, self.model)

        if self.gpu >= 0:
            cuda.get_device(self.gpu).use()
            self.model.to_gpu()

    def load_image(self, imagefile):
        img = io.imread(imagefile)
        return img

    def __call__(self, inputs):
        """Applies inference computation to input arrays.

        Args:
            inputs: input arrays.
        Return:
            dictionary: Dictionary of :class: objects.
        """
        raise NotImplementedError()
