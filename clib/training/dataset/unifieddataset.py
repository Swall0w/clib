import numpy
from .basedataset import BaseLabeledImageDataset
from skimage import io, transform, color


class UnifiedLabeledImageDataset(BaseLabeledImageDataset):
    def get_example(self, i):
        full_path, int_label = self._pairs[i]
        image = io.imread(full_path)
        if self.resize:
            image = transform.resize(image, (self.resize, self.resize), mode='reflect')
        if image.ndim == 2:
            image = color.gray2rgb
        label = numpy.array(int_label, dtype=self._label_dtype)
        return image.transpose(2, 0, 1), label
