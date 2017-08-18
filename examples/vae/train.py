#!/usr/bin/env python

import argparse
import os

import chainer
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from chainer import cuda, training
from chainer.training import Trainer, extensions

import net

matplotlib.use('Agg')


def save_image(x, filename):
    fig, ax = plt.subplots(3, 3, figsize=(9, 9), dpi=100)
    for ai, xi in zip(ax.flatten(), x):
        ai.imshow(xi.reshape(28, 28))
    fig.savefig(filename)
    plt.close('all')

def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the optimization from snapshot')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch', '-e', default=20, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--dimz', '-z', default=20, type=int,
                        help='dimention of encoded vector')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='learning minibatch size')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.add_argument('--out', '-o', type=str, default='./result/',
                        help='dir to save snapshots.')
    parser.add_argument('--interval', '-i', type=int, default=5, help='interval of save images.')
    parser.add_argument
    args = parser.parse_args()

    batchsize = args.batchsize
    n_epoch = args.epoch
    n_latent = args.dimz

    print('GPU: {}'.format(args.gpu))
    print('# dim z: {}'.format(args.dimz))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    model = net.VAE(784, n_latent, 500)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()
    xp = np if args.gpu < 0 else cuda.cupy

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train, test = chainer.datasets.get_mnist(withlabel=False)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    if not os.path.exists(os.path.join(args.out, 'cg.dot')):
        print('dump computational graph of `main/loss`')
        trainer.extend(extensions.dump_graph('main/loss'))

    trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}'), trigger=(args.interval, 'epoch'))

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss']))
    trainer.extend(extensions.ProgressBar())

    @training.make_extension(trigger=(args.interval, 'epoch'))
    def save_images(trainer):
        out_dir = os.path.join(trainer.out, 'epoch_{}'.format(str(trainer.updater.epoch)))
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        train_ind = [1, 3, 5, 10, 2, 0, 13, 15, 17]
        x = chainer.Variable(np.asarray(train[train_ind]), volatile='on')
        x1 = model.decode(model.encode(x)[0])
        save_image(x.data, filename=os.path.join(out_dir,'train'))
        save_image(x1.data, filename=os.path.join(out_dir, 'train_reconstructed'))

        test_ind = [3, 2, 1, 18, 4, 8, 11, 17, 61]
        x = chainer.Variable(np.asarray(test[test_ind]), volatile='on')
        x1 = model(x)
        x1 = model.decode(model.encode(x)[0])
        save_image(x.data, filename=os.path.join(out_dir, 'test'))
        save_image(x1.data, filename=os.path.join(out_dir, 'test_reconstructed'))

        z = chainer.Variable(np.random.normal(0, 1, (9, n_latent)).astype(np.float32))
        x = model.decode(z)
        save_image(x.data, filename=os.path.join(out_dir, 'sampled'))
    trainer.extend(save_images)

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
