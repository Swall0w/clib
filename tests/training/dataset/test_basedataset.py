import unittest
from clib.training.dataset import BaseLabeledImageDataset 


class BaseDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tagfile = 'tests/data/voc.names'
        with open(self.tagfile, 'r') as f:
            self.tags = { item.strip(): int(x) for x, item in enumerate(f.readlines())}
        self.labelfile = 'tests/data/label.txt'
        self.dataset = BaseLabeledImageDataset(pairs=self.labelfile,
                                              label_dict=self.tags)

    def test_basedataset(self):
        self.assertEqual(len(self.dataset), 1)
