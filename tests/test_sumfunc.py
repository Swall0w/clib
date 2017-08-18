import unittest

from clib.tmp import sumfunc

from .context import clib


class SumFuncTest(unittest.TestCase):
    def test_sumfunc(self):
        self.assertEqual(sumfunc(2), 3)

if __name__ == '__main__':
    unittest.main()
