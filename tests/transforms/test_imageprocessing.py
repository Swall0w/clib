import unittest

from clib.transforms import (add_noise, add_salt_and_pepper_noise, brightness,
                             contrast, elastic_transform, gamma_adjust,
                             gaussian_blur, saturation, sharpness)
from skimage import data, io


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

    def test_brightness(self):
        self.assertEqual(brightness(self.grayimg).shape,
                         (303, 384))
        self.assertEqual(brightness(self.rgbimg).shape,
                         (512, 512, 3))


class SaturationTest(unittest.TestCase):
    def setUp(self):
        self.grayimg = data.coins()
        self.rgbimg = data.astronaut()

    def test_saturation(self):
        self.assertEqual(saturation(self.grayimg).shape,
                         (303, 384))
        self.assertEqual(saturation(self.rgbimg).shape,
                         (512, 512, 3))


class SharpnessTest(unittest.TestCase):
    def setUp(self):
        self.grayimg = data.coins()
        self.rgbimg = data.astronaut()

    def test_sharpness(self):
        self.assertEqual(sharpness(self.grayimg).shape,
                         (303, 384))
        self.assertEqual(sharpness(self.rgbimg).shape,
                         (512, 512, 3))


class GammaAdjustTest(unittest.TestCase):
    def setUp(self):
        self.grayimg = data.coins()
        self.rgbimg = data.astronaut()

    def test_gamma_adjust(self):
        self.assertEqual(gamma_adjust(self.grayimg).shape,
                         (303, 384))
        self.assertEqual(gamma_adjust(self.rgbimg).shape,
                         (512, 512, 3))
