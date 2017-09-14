import numpy
import six
from chainer.dataset import dataset_mixin
from clib.utils.regrex import is_path


class BaseLabeledImageDataset(dataset_mixin.DatasetMixin):

    def __init__(self, pairs, label_dict, dtype=numpy.float32,
                 label_dtype=numpy.int32, resize=None, random_step=0,
                 is_image_aug=False):
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
        raise NotImplementedError()
