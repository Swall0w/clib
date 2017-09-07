import chainer
import chainer.functions as F
import chainer.links as L


class WGANGenerator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__(
            fc1=L.Linear(None, 800),
            fc2=L.Linear(None, 28*28)
        )

    def __call__(self, z):
        h = F.relu(self.fc1(z))
        y = F.reshape(F.sigmoid(self.fc2(h)), (-1, 1, 28, 28))
        return y


class WGANCritic(chainer.Chain):
    def __init__(self):
        super(Critic, self).__init__(
            fc1=L.Linear(None, 800),
            fc2=L.Linear(None, 28*28)
        )

    def __call__(self, x):
        batchsize = x.shape[0]
        h = F.relu(self.fc1(x))
        y = F.sum(self.fc2(h)) / batchsize
        return y
