import unittest
from clib.training.dataset import XMLLabeledImageDataset
from clib.utils import load_class


class XMLDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tagfile = 'tests/data/voc.names'
        self.labelfile = 'tests/data/label.txt'
        self.dataset = XMLLabeledImageDataset(pairs=self.labelfile,
                                              label_dict=self.tagfile,
                                              resize=(10, 10),
                                              is_image_aug=True)

    def test_xmldataset(self):
        self.assertEqual(len(self.dataset), 1)
        self.assertEqual(len(self.dataset[0]), 2)
