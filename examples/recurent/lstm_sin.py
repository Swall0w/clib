import argparse

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset


class SimpleLSTM(chainer.Chain):
    def __init__(self, n_input=1, n_units=5, n_output=1):
        super(SimpleLSTM, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(n_input, n_units)
            self.l2 = L.LSTM(n_units, n_units)
            self.l3 = L.LSTM(n_units, n_units)
            self.l4 = L.Linear(n_units, n_output)

#        for param in self.params():
#            param.data[...] = np.random.uniform(-0.1,
#                                                0.1,
#                                                param.data.shape)

    def reset_state(self):
        self.l2.reset_state()
        self.l3.reset_state()

    def __call__(self, x):
        h1 = self.l1(F.dropout(x))
        h2 = self.l2(F.dropout(h1))
        h3 = self.l3(F.dropout(h2))
        y = self.l4(F.dropout(h3))
        return y


class LossFunc(chainer.Chain):
    def __init__(self, predictor):
        super(LossFunc, self).__init__(predictor=predictor)

    def __call__(self, x, t):
#        x.data = x.data.reshape((-1, 1)).astype(np.float32)
#        t.data = t.data.reshape((-1, 1)).astype(np.float32)

        y = self.predictor(x)
        loss = F.mean_squared_error(y, t)
        chainer.report({'loss':loss}, self)
        return loss

class ParallelSequentialIterator(chainer.dataset.Iterator):

    def __init__(self, dataset, batch_size, repeat=True):
        self.dataset = dataset
        self.batch_size = batch_size  # batch size
        self.epoch = 0
        self.is_new_epoch = False
        self.repeat = repeat
        length = len(dataset)
        self.offsets = [i * length // batch_size for i in range(batch_size)]
        self.iteration = 0
        self._previous_epoch_detail = -1.

    def __next__(self):
        # This iterator returns a list representing a mini-batch. Each item
        # indicates a different position in the original sequence. Each item is
        # represented by a pair of two word IDs. The first word is at the
        # "current" position, while the second word at the next position.
        # At each iteration, the iteration count is incremented, which pushes
        # forward the "current" position.
        length = len(self.dataset)
        if not self.repeat and self.iteration * self.batch_size >= length:
            # If not self.repeat, this iterator stops at the end of the first
            # epoch (i.e., when all words are visited once).
            raise StopIteration
        cur_words = self.get_words()
        self._previous_epoch_detail = self.epoch_detail
        self.iteration += 1
        next_words = self.get_words()
#        print('cur: ',cur_words, ' next: ', next_words)

        epoch = self.iteration * self.batch_size // length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch

#        return list(zip(cur_words, next_words))
        ret = list(zip(cur_words, next_words))
#        print('ret: ',ret)
        return ret


    @property
    def epoch_detail(self):
        # Floating point version of epoch.
        return self.iteration * self.batch_size / len(self.dataset)

    @property
    def previous_epoch_detail(self):
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail

    def get_words(self):
        # It returns a list of current words.
        return [self.dataset[(offset + self.iteration) % len(self.dataset)]
                for offset in self.offsets]

    def serialize(self, serializer):
        # It is important to serialize the state to be recovered on resume.
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)
        try:
            self._previous_epoch_detail = serializer(
                'previous_epoch_detail', self._previous_epoch_detail)
        except KeyError:
            # guess previous_epoch_detail for older version
            self._previous_epoch_detail = self.epoch + \
                (self.current_position - self.batch_size) / len(self.dataset)
            if self.epoch_detail > 0:
                self._previous_epoch_detail = max(
                    self._previous_epoch_detail, 0.)
            else:
                self._previous_epoch_detail = -1.


# Custom updater for truncated BackProp Through Time (BPTT)
class BPTTUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, bprop_len, device):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, device=device)
        self.bprop_len = bprop_len

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        loss = 0
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Progress the dataset iterator for bprop_len words at each iteration.
        for i in range(self.bprop_len):
            # Get the next batch (a list of tuples of two word IDs)
            batch = train_iter.__next__()

            # Concatenate the word IDs to matrices and send them to the device
            # self.converter does this job
            # (it is chainer.dataset.concat_examples by default)
            x, t = self.converter(batch, self.device)
#            print('x: ',x, 't: ',t )

            # Compute the loss at this time step and accumulate it
            loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))

        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters


# Routine to rewrite the result dictionary of LogReport to add perplexity
# values
def compute_perplexity(result):
    result['perplexity'] = np.exp(result['main/loss'])
    if 'validation/main/loss' in result:
        result['val_perplexity'] = np.exp(result['validation/main/loss'])




def sin_data(N_data=100, N_loop=10):
    t = np.linspace(0., 2*np.pi*N_loop, num=N_data)

    X = 0.8 * np.sin(2.0*t)

    N_train = int(N_data*0.8)
    N_test = int(N_data*0.2)

    tmp_Dataset_x = np.array(X).astype(np.float32)
    x_train, x_test = np.array(tmp_Dataset_x[:N_train]), np.array(tmp_Dataset_x[N_train:])
#    train = tuple_dataset.TupleDataset(x_train, t[:N_train])
#    test = tuple_dataset.TupleDataset(x_test, t[N_train:])
    train = tuple_dataset.TupleDataset(x_train)
    test = tuple_dataset.TupleDataset(x_test)
    return train, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=20,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--bproplen', '-l', type=int, default=35,
                        help='Number of words in each mini-batch '
                             '(= length of truncated BPTT)')
    parser.add_argument('--epoch', '-e', type=int, default=39,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.set_defaults(test=False)
    parser.add_argument('--unit', '-u', type=int, default=650,
                        help='Number of LSTM units in each layer')
    parser.add_argument('--model', '-m', default='model.npz',
                        help='Model file name to serialize')
    args = parser.parse_args()


    train, test = sin_data()
    train_iter = ParallelSequentialIterator(train, args.batchsize)


    # Prepare an RNNLM model
    model = LossFunc(SimpleLSTM())
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # Set up an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

    # Set up a trainer
    updater = BPTTUpdater(train_iter, optimizer, args.bproplen, args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)


    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss']
    ))

    trainer.run()



if __name__ == '__main__':
    main()
