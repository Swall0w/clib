import unittest
from clib.training.dataset import UnifiedLabeledImageDataset


class UnifiedDatasetTest(unittest.TestCase):
    def setUp(self):
        self.labelfile = 'tests/data/label_class.txt'
        self.dataset = UnifiedLabeledImageDataset(pairs=self.labelfile)

    def test_xmldataset(self):
        self.assertEqual(len(self.dataset), 1)
        self.assertEqual(len(self.dataset[0]), 2)
        self.assertEqual(self.dataset[0][1], 0)
