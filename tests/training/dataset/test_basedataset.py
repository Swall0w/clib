import unittest
from clib.training.dataset import BaseLabeledImageDataset 
from clib.utils import load_class


class BaseDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tagfile = 'tests/data/voc.names'
#        with open(self.tagfile, 'r') as f:
#            self.tags = { item.strip(): int(x) for x, item in enumerate(f.readlines())}
#        self.tags = load_class(self.tagfile)
        self.labelfile = 'tests/data/label.txt'
        self.dataset = BaseLabeledImageDataset(pairs=self.labelfile,
                                              label_dict=self.tagfile)

    def test_basedataset(self):
        self.assertEqual(len(self.dataset), 1)
