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

    def test_blur(self):
        self.assertEqual(self.imag.blur(img=self.grayimg, israndom=True).shape,
                         (303, 384))
        self.assertEqual(self.imag.blur(img=self.rgbimg, israndom=True).shape,
                         (512, 512, 3))
        self.assertEqual(self.imag.blur(img=self.grayimg).shape,
                         (303, 384))
        self.assertEqual(self.imag.blur(img=self.rgbimg).shape,
                         (512, 512, 3))

    def test_noise(self):
        self.assertEqual(self.imag.noise(img=self.grayimg, israndom=True).shape,
                         (303, 384))
        self.assertEqual(self.imag.noise(img=self.rgbimg, israndom=True).shape,
                         (512, 512, 3))
        self.assertEqual(self.imag.noise(img=self.grayimg).shape,
                         (303, 384))
        self.assertEqual(self.imag.noise(img=self.rgbimg).shape,
                         (512, 512, 3))

    def test_spnoise(self):
        self.assertEqual(self.imag.sp_noise(img=self.grayimg, israndom=True).shape,
                         (303, 384))
        self.assertEqual(self.imag.sp_noise(img=self.rgbimg, israndom=True).shape,
                         (512, 512, 3))
        self.assertEqual(self.imag.sp_noise(img=self.grayimg).shape,
                         (303, 384))
        self.assertEqual(self.imag.sp_noise(img=self.rgbimg).shape,
                         (512, 512, 3))

    def test_contrast(self):
        self.assertEqual(self.imag.contrast(img=self.grayimg, israndom=True).shape,
                         (303, 384))
        self.assertEqual(self.imag.contrast(img=self.rgbimg, israndom=True).shape,
                         (512, 512, 3))
        self.assertEqual(self.imag.contrast(img=self.grayimg).shape,
                         (303, 384))
        self.assertEqual(self.imag.contrast(img=self.rgbimg).shape,
                         (512, 512, 3))

    def test_brightness(self):
        self.assertEqual(self.imag.brightness(img=self.grayimg, israndom=True).shape,
                         (303, 384))
        self.assertEqual(self.imag.brightness(img=self.rgbimg, israndom=True).shape,
                         (512, 512, 3))
        self.assertEqual(self.imag.brightness(img=self.grayimg).shape,
                         (303, 384))
        self.assertEqual(self.imag.brightness(img=self.rgbimg).shape,
                         (512, 512, 3))

    def test_saturation(self):
        self.assertEqual(self.imag.saturation(img=self.grayimg, israndom=True).shape,
                         (303, 384))
        self.assertEqual(self.imag.saturation(img=self.rgbimg, israndom=True).shape,
                         (512, 512, 3))
        self.assertEqual(self.imag.saturation(img=self.grayimg).shape,
                         (303, 384))
        self.assertEqual(self.imag.saturation(img=self.rgbimg).shape,
                         (512, 512, 3))

    def test_sharpness(self):
        self.assertEqual(self.imag.sharpness(img=self.grayimg, israndom=True).shape,
                         (303, 384))
        self.assertEqual(self.imag.sharpness(img=self.rgbimg, israndom=True).shape,
                         (512, 512, 3))
        self.assertEqual(self.imag.sharpness(img=self.grayimg).shape,
                         (303, 384))
        self.assertEqual(self.imag.sharpness(img=self.rgbimg).shape,
                         (512, 512, 3))

    def test_gamma_adjust(self):
        self.assertEqual(self.imag.gamma_adjust(img=self.grayimg, israndom=True).shape,
                         (303, 384))
        self.assertEqual(self.imag.gamma_adjust(img=self.rgbimg, israndom=True).shape,
                         (512, 512, 3))
        self.assertEqual(self.imag.gamma_adjust(img=self.grayimg).shape,
                         (303, 384))
        self.assertEqual(self.imag.gamma_adjust(img=self.rgbimg).shape,
                         (512, 512, 3))

    def test_crop(self):
        self.assertEqual(self.imag.crop(self.rgbimg, (0, 50, 100, 200)).shape,
                         (100, 150, 3))
        self.assertEqual(self.imag.crop(self.grayimg, (0, 50, 100, 200)).shape,
                         (100, 150))

    def test_resize(self):
        self.assertEqual(self.imag.resize(self.rgbimg, (100, 50)).shape, (100, 50, 3))
        self.assertEqual(self.imag.resize(self.rgbimg[20:50, 100:200, :],
                                          (100, 50)).shape, (100, 50, 3))
