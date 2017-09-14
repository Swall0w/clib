import unittest

from clib.utils import randombool

class RandomBoolTest(unittest.TestCase):
    def test_randombool(self):
        self.assertIsInstance(randombool(), bool)
