import numpy as np
import chainer
import chainer.links as L
from chainer import Chain, Variable
from chainer import datasets

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

def main():
    pass

if __name__ == '__main__':
    main()
