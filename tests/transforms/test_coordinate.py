import unittest

from clib.transforms import _check_position, jitter_position
from skimage import data, io


class JitterTest(unittest.TestCase):
    def setUp(self):
        self.grayimg = data.coins()
        self.rgbimg = data.astronaut()
        self.bbox = (20, 40, 60, 80)
        self.grayimg_size = (303, 384)
        self.rgbimg_size = (512, 512)

    def test_jitter_position(self):
        self.assertEqual(jitter_position(self.bbox, self.grayimg.shape),
                         (20, 40, 60, 80))

class CheckPositionTest(unittest.TestCase):
    def setUp(self):
        self.bbox1t = (20, 15, 30, 20)
        self.bbox2f = (30, 15, 30, 20)
        self.bbox3f = (20, 20, 30, 20)
        self.bbox4f = (40, 20, 30, 21)
        self.bbox5f = (20, 30, 30, 20)

    def test_check_position(self):
        self.assertEqual(_check_position(self.bbox1t), True)
        self.assertEqual(_check_position(self.bbox2f), False)
        self.assertEqual(_check_position(self.bbox3f), False)
        self.assertEqual(_check_position(self.bbox4f), False)
        self.assertEqual(_check_position(self.bbox5f), False)
