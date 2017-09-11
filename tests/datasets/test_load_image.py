import unittest
from clib.datasets import ImageAugmentation
from skimage import  data, io


class ImageAugmentationTest(unittest.TestCase):
    def setUp(self):
        self.grayimg = data.coins()
        self.rgbimg = data.astronaut()
        self.imgpath = 'tests/data/0.jpg'
        self.imag = ImageAugmentation()

    def test_read(self):
        self.assertEqual(self.imag.read(self.imgpath).ndim, 3)

    def test_noise(self):
        self.assertEqual(self.imag.noise(img=self.grayimg).shape,
                         (303, 384))
        self.assertEqual(self.imag.noise(img=self.rgbimg).shape,
                         (512, 512, 3))
