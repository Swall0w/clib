import unittest
from clib.transforms import (elastic_transform,
                             gaussian_blur,
                             add_noise,
                             add_salt_and_pepper_noise,
                             contrast, brightness)
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


class AddNoiseTest(unittest.TestCase):
    def setUp(self):
        self.grayimg = data.coins()
        self.rgbimg = data.astronaut()

    def test_add_noise(self):
        self.assertEqual(add_noise(self.grayimg).shape,
                         (303, 384))
        self.assertEqual(add_noise(self.rgbimg).shape,
                         (512, 512, 3))


class AddSPNoiseTest(unittest.TestCase):
    def setUp(self):
        self.grayimg = data.coins()
        self.rgbimg = data.astronaut()

    def test_add_spnoise(self):
        self.assertEqual(add_salt_and_pepper_noise(self.grayimg).shape,
                         (303, 384))
        self.assertEqual(add_salt_and_pepper_noise(self.rgbimg).shape,
                         (512, 512, 3))


class ContrastTest(unittest.TestCase):
    def setUp(self):
        self.grayimg = data.coins()
        self.rgbimg = data.astronaut()

    def test_contrast(self):
        self.assertEqual(contrast(self.grayimg).shape,
                         (303, 384))
        self.assertEqual(contrast(self.rgbimg).shape,
                         (512, 512, 3))


class BrightnessTest(unittest.TestCase):
    def setUp(self):
        self.grayimg = data.coins()
        self.rgbimg = data.astronaut()

    def test_contrast(self):
        self.assertEqual(brightness(self.grayimg).shape,
                         (303, 384))
        self.assertEqual(brightness(self.rgbimg).shape,
                         (512, 512, 3))
