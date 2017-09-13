import os
import unittest

import clib
from clib.utils import Box


class BboxTest(unittest.TestCase):
    def setUp(self):
        self.bbox = Box(50, 50, 40, 60)

    def test_vi_bbox(self):
        self.assertEqual(self.bbox.int_left_top(), (30, 20))
        self.assertEqual(self.bbox.int_right_bottom(), (70, 80))
        self.assertEqual(self.bbox.left_top(), [30.0, 20.0])
        self.assertEqual(self.bbox.right_bottom(), [70.0, 80.0])
        self.bbox.crop_region(5, 5)
        self.assertEqual(self.bbox.right_bottom(), [5.0, 5.0])

if __name__ == '__main__':
    unittest.main()
