import unittest
from clib.converts import resize_to_yolo, batch
from skimage import data
from skimage.transform import resize


class ResizeYoloTest(unittest.TestCase):
    def setUp(self):
        self.grayimg = data.coins()
        self.rgbimg = data.astronaut()

    def test_resize_to_yolo(self):
        self.assertEqual(resize_to_yolo(self.rgbimg).shape,
                         (448, 448, 3))
#        with self.assertRaises(ValueError ) as cm:
#            resize_to_yolo(self.grayimg)
#        exception = cm.exception
#        self.assertEqual(exception.message, '')


class BatchTest(unittest.TestCase):
    def setUp(self):
        self.rgbimg1 = data.astronaut()
        self.rgbimg2 = resize(self.rgbimg1, (400, 400), mode='reflect')

        self.rgbimg1 = self.rgbimg1.transpose(2, 0, 1)
        self.rgbimg2 = self.rgbimg2.transpose(2, 0, 1)

        self.dat = [(self.rgbimg2, 0),
                    (self.rgbimg1, 1)]

    def test_batch(self):
        self.assertEqual(batch(self.dat)[1][0].shape,
                         (3, 400, 400))
