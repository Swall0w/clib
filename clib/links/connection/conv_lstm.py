import chainer.functions as F
import chainer.links as L
from chainer import Chain, Variable

import cupy


class ConvLSTM(Chain):
    """ Convolutional LSTM unit.

    This is a Convolutional LSTM unit as a chain.

    https://arxiv.org/pdf/1506.04214.pdf
    """
    def __init__(self, channelIn=1, channelOut=1, ksize=5):
        padsize = (ksize - 1) / 2
        self.channelIn = channelIn
        self.channelOut = channelOut
        self.ksize = ksize
        self.padsize = padsize
        self.stride = 1

        super(ConvLSTM, self).__init__(
            Wz=L.Convolution2D(in_channels=self.channelIn,
                               out_channels=self.channelOut,
                               ksize=self.ksize, stride=self.stride,
                               pad=self.padsize),
            Wi=L.Convolution2D(in_channels=self.channelIn,
                               out_channels=self.channelOut,
                               ksize=self.ksize, stride=self.stride,
                               pad=self.padsize),
            Wf=L.Convolution2D(in_channels=self.channelIn,
                               out_channels=self.channelOut,
                               ksize=self.ksize, stride=self.stride,
                               pad=self.padsize),
            Wo=L.Convolution2D(in_channels=self.channelIn,
                               out_channels=self.channelOut,
                               ksize=self.ksize, stride=self.stride,
                               pad=self.padsize),
            Rz=L.Convolution2D(in_channels=self.channelIn,
                               out_channels=self.channelOut,
                               ksize=self.ksize, stride=self.stride,
                               pad=self.padsize),
            Ri=L.Convolution2D(in_channels=self.channelIn,
                               out_channels=self.channelOut,
                               ksize=self.ksize, stride=self.stride,
                               pad=self.padsize),
            Rf=L.Convolution2D(in_channels=self.channelIn,
                               out_channels=self.channelOut,
                               ksize=self.ksize, stride=self.stride,
                               pad=self.padsize),
            Ro=L.Convolution2D(in_channels=self.channelIn,
                               out_channels=self.channelOut,
                               ksize=self.ksize, stride=self.stride,
                               pad=self.padsize),
        )

    def __call__(self, s):
        # s is expected to cupyArray(num, height, width)
        accum_loss = None
        chan = self.channelIn
        hei = len(s[0])
        wid = len(s[0][0])
        h = Variable(cupy.zeros((1, chan, hei, wid), dtype=cupy.float32))
        c = Variable(cupy.zeros((1, chan, hei, wid), dtype=cupy.float32))

        for i in range(len(s) - 1):
            # len(s) is expected to 26

            tx = Variable(cupy.array(
                s[i + 1], dtype=cupy.float32).reshape(1, chan, hei, wid))
            x_k = Variable(cupy.array(
                s[i], dtype=cupy.float32).reshape(1, chan, hei, wid))
            z0 = self.Wz(x_k) + self.Rz(h)
            z1 = F.tanh(z0)
            i0 = self.Wi(x_k) + self.Ri(h)
            i1 = F.sigmoid(i0)
            f0 = self.Wf(x_k) + self.Rf(h)
            f1 = F.sigmoid(f0)
            c = z1 * i1 + f1 * c
            o0 = self.Wo(x_k) + self.Ro(h)
            o1 = F.sigmoid(o0)
            h = o1 * F.tanh(c)
            loss = F.mean_squared_error(h, tx)
            accum_loss = loss if accum_loss is None else accum_loss + loss

        return accum_loss
