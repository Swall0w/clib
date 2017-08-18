import argparse

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions

from alex import Alex


def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in eeach mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    return parser.parse_args()


def main():
    args = arg()
    model = L.Classifier(Alex())
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    
    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    updater = training.StandardUpdater()


if __name__ == '__main__':
    main()
