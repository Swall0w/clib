import unittest
from clib.training.dataset import YoloPreprocessedDataset
from clib.utils import load_class


class YoloDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tagfile = 'tests/data/voc.names'
        self.labelfile = 'tests/data/label_yolo.txt'
        self.dataset1 = YoloPreprocessedDataset(pairs=self.labelfile,
                                                label_dict=self.tagfile,
                                                resize=[100, 110])
        self.dataset2 = YoloPreprocessedDataset(pairs=self.labelfile,
                                                label_dict=self.tagfile,
                                                resize=224)

    def test_xmldataset(self):
        self.assertEqual(len(self.dataset1), 1)
        self.assertEqual(len(self.dataset1[0]), 2)
        self.assertEqual(len(self.dataset2), 1)
        self.assertEqual(len(self.dataset2[0]), 2)
