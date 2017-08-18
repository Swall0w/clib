import unittest

import chainer
import numpy
from chainer import testing


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))


class TestSumofSquared(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)

    def check_forward(self,x):
        print(x)

    def test_forward_cpu(self):
        self.check_forward(self.x)

testing.run_module(__name__, __file__)
