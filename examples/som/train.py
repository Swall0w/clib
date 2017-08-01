import numpy as np
import chainer
import chainer.links as L
from chainer import Chain, Variable
from chainer import datasets
import cv2

class SOM(Chain):
    def __init__(self, width):
        self.width = width
        super(SOM,self).__init__(
            competitive = L.Linear(None,self.width*self.width, nobias=True)
        )
    def predict(self,x):
        ip = self.competitive(x)
        pos = np.argmax(ip.data)
        return pos/self.width, pos%self.width
    def __neighbor(self, center, var):
        y = np.abs(np.arange(-center[0], self.width - center[0]))
        x = np.abs(np.arange(-center[1], self.width - center[1]))
        xx, yy = np.meshgrid(x,y)
        d2 = xx**2 + yy**2
        return np.exp(-d2/(2*(var**2)))

    def incr_learn(self, x, lr=0.1, var=4.0, reinforce=True):
        pos = self.predict(x)
        delta_x = (lr * self.__neighbor(pos, var).reshape(1, -1)).T.dot(x.data)
        delta_w = (lr * self.__neighbor(pos, var).reshape(1, -1)).T * self.competitive.W.data
        self.competitive.W.data += delta_x if reinforce else - delta_x
        self.competitive.W.data -= delta_w if reinforce else - delta_w

    def weight_show(self, in_width, ch):
        show_array = np.zeros((in_width * self.width, in_width*self.width, ch),dtype=np.float32)
        for i, c in enumerate(self.competitive.W.data):
            y = i // self.width
            x = i % self.width
            if ch == 3:
                show_array[y*in_width:(y+1)*in_width, x*in_width:(x+1)*in_width] = cv2.cvtColor(np.rollaxis(c.reshape(ch, in_width, in_width), 0, 3), cv2.COLOR_RGB2BGR)
            else:
                show_array[y*in_width:(y+1)*in_width, x*in_width:(x+1)*in_width] = c.reshape(in_width, in_width,1)
        cv2.imshow('win', show_array)
        cv2.waitKey(1)


def main():
    som = SOM(width=10)
    train, test = datasets.get_mnist()

    for it, tr in enumerate(train):
        if it % 5000 == 0:
            print('Iteration: {0}'.format(it))

        x = Variable(np.array([tr[0]], dtype=np.float32))

        lr = 0.05 * (1.0 - float(it) / len(train))
        var = 2.0 * (1.0 - float(it) / len(train))

        som.incr_learn(x, lr=lr, var=var)

        som.weight_show(in_width=28, ch =1)

if __name__ == '__main__':
    main()
