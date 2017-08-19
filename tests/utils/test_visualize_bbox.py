import os
import unittest

import clib
from clib.utils import viz_bbox
from PIL import Image


class VisBboxTest(unittest.TestCase):
    def setUp(self):
        self.img = Image.open(os.path.abspath('./tests/data/0.jpg'))
        self.output = [{"class": "person",
                        "left": 174,
                        "right": 349,
                        "top": 101,
                        "bottom": 351,
                        "prob": 1}]

    def test_vi_bbox(self):
        self.assertIsInstance(viz_bbox(self.img, self.output), Image.Image)

if __name__ == '__main__':
    unittest.main()
