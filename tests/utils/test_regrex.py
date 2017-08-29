import unittest

from clib.utils.regrex import is_path


class IsPath(unittest.TestCase):
    def test_is_path(self):
        self.assertTrue(is_path('/home/hoge/test.xml'))
        self.assertTrue(is_path('../hoge/fuga.txt'))
        self.assertTrue(is_path('./cname/test.xml'))
        self.assertFalse(is_path('0'))
        self.assertFalse(is_path('123'))
        self.assertFalse(is_path('yomeyome'))

if __name__ == '__main__':
    unittest.main()
