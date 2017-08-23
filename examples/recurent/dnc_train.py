from clib.links.model.recurrent import DNC

import numpy as np
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers, Chain, Link, Variable


def onehot(x, n):
    ret = np.zeros(n).astype(np.float32)
    ret[x] = 1.0
    return ret


def main():
    X = 5
    Y = 5
    N = 10
    W = 10
    R = 2


    model = DNC(X, Y, N, W, R)
    optimizer = optimizers.Adam()
    optimizer.setup(model)


    n_data = 10000 # number of input data
    loss = 0.0
    acc = 0.0
    acc_bool = []


    for data_cnt in range(n_data):

        loss_frac = np.zeros((1, 2))    

# prepare one pair of input and target data 
# length of input data is randomly set
        len_content = np.random.randint(3, 6) 
# generate one input data as a sequence of randam integers
        content = np.random.randint(0, X-1, len_content) 
        len_seq = len_content + len_content # the former is for input, the latter for the target
        x_seq_list = [float('nan')] * len_seq # input sequence
        t_seq_list = [float('nan')] * len_seq # target sequence

        for i in range(len_seq):
# convert a format of input data
            if (i < len_content):
                x_seq_list[i] = onehot(content[i], X)
            elif (i == len_content):
                x_seq_list[i] = onehot(X-1, X)
            else:
                x_seq_list[i] = np.zeros(X).astype(np.float32)
# convert a format of output data
            if (i >= len_content):
                t_seq_list[i] = onehot(content[i - len_content], X)         

        model.reset_state() # reset reccurent state per input data

# input data is fed as a sequence
        for cnt in range(len_seq):
            x = Variable(x_seq_list[cnt].reshape(1, X))
            if (isinstance(t_seq_list[cnt], np.ndarray)):
                t = Variable(t_seq_list[cnt].reshape(1, Y))
            else:
                t = []

            y = model(x)

            if (isinstance(t, chainer.Variable)):
                loss += (y - t)**2
                acc_bool.append(np.argmax(y.data)==np.argmax(t.data))                        
                if (np.argmax(y.data)==np.argmax(t.data)): acc += 1

            if (cnt+1==len_seq):
# training by back propagation
                model.cleargrads()
                loss.grad = np.ones(loss.shape, dtype=np.float32)
                loss.backward()
                optimizer.update()
                loss.unchain_backward()
# print loss and accuracy
                if data_cnt < 50 or data_cnt >= 9950:
                    print('(', data_cnt, ')', acc_bool, ' :: ', loss.data.sum()/loss.data.size/len_content, ' :: ', acc/len_content)
                loss_frac += [loss.data.sum()/loss.data.size/len_seq, 1.]
                loss = 0.0
                acc = 0.0
                acc_bool = []

if __name__ == '__main__':
    main()
