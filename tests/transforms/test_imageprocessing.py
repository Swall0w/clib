import unittest
from clib.transforms import (elastic_transform,
                             gaussian_blur)
from skimage import io, data


class ElasticTransformTest(unittest.TestCase):
    def setUp(self):
        self.grayimg = data.coins()
        self.rgbimg = data.astronaut()

    def test_elastic_transform(self):
        self.assertEqual(elastic_transform(self.grayimg).shape,
                         (303, 384))
        self.assertEqual(elastic_transform(self.rgbimg).shape,
                         (512, 512, 3))

class GaussianBlurTest(unittest.TestCase):
    def setUp(self):
        self.grayimg = data.coins()
        self.rgbimg = data.astronaut()

    def test_gaussian_blur(self):
        self.assertEqual(gaussian_blur(self.grayimg).shape,
                         (303, 384))
        self.assertEqual(gaussian_blur(self.rgbimg).shape,
                         (512, 512, 3))
