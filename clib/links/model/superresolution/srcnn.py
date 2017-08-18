import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter


class SRCNN(chainer.Chain):
    def __init__(self):
        super(SRCNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, 9)
            self.conv2 = L.Convolution2D(64, 32, 3)
            self.conv3 = L.Convolution2D(32, 3, 5)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        return h


class SRCNNPredictor(chainer.Chain):
    def __init__(self, predictor,
                 lossfunc=F.mean_squared_error):
        super(SRCNNPredictor, self).__init__()
        self.lossfunc = lossfunc
        self.y = None
        self.loss = None
        self.accuracy = None

        with self.init_scope():
            self.predictor = predictor

    def __call__(self, x, t):
        self.y = None
        self.loss = None
        self.accuracy = None
        self.y = self.predictor(x)

        self.loss = self.lossfunc(self.y, t)
        reporter.report({'loss': self.loss}, self)
        return self.loss
