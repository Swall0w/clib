import unittest
from clib.converts import resize_to_yolo
from skimage import data


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
