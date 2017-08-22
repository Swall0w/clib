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

if __name__ == '__main__':
    unittest.main()
