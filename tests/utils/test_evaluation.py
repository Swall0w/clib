import unittest
from collections import namedtuple
from clib.utils import bb_intersection_over_union


detection = namedtuple("detection", ['image_name', 'gt', 'pred'])

class IOUTEST(unittest.TestCase):
    def setUp(self):
        self.example = detection('img.jpg', [39, 63, 203, 112],
                                 [54, 66, 198, 114])

    def test_bbox_iou(self):
        self.assertEqual(bb_intersection_over_union(
            self.example.gt, self.example.pred), 0.7980093676814989)
