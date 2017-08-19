import argparse
import os
import random

import chainer
import numpy as np
from chainer import training
from chainer.training import extensions
from clib.links.model.superresolution.srcnn import SRCNN, SRCNNPredictor


def arg():
    parser = argparse.ArgumentParser(
        description='SRCNN trainer')
    parser.add_argument('--train',
                        help='Path to training image-label list file')
    parser.add_argument('--val',
                        help='Path to validation image-label list file')
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU')
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--loaderjob', '-j', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                        help='Validation minibatch size')
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    return parser.parse_args()


class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, crop_size, random=True):
        self.base = chainer.datasets.ImageDataset(path)
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        crop_size = self.crop_size

        image = self.base[i]
        _, h, w = image.shape

        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        image *= (1.0 / 255.0)  # Scale to [0, 1]
        return image, image


def read_dataset(pathfile):
    if os.path.isfile(pathfile):
        with open(pathfile, 'r') as f:
            datalist = [line.strip() for line in f.readlines()]
        return datalist
    elif isinstance(pathfile, list):
        return pathfile


def main():
    args = arg()

    model = SRCNNPredictor(SRCNN())

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()  # Make the GPU current
        model.to_gpu()

    cropsize = 64
    train = PreprocessedDataset(read_dataset(args.train), args.root, cropsize)
    val = PreprocessedDataset(
        read_dataset(args.val), args.root, cropsize, False)

    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize, n_processes=args.loaderjob)
    val_iter = chainer.iterators.MultiprocessIterator(
        val, args.val_batchsize, repeat=False, n_processes=args.loaderjob)

    # Set up an optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    val_interval = (10 if args.test else 100000), 'iteration'
    log_interval = (10 if args.test else 1000), 'iteration'

    trainer.extend(extensions.Evaluator(val_iter, model, device=args.gpu),
                   trigger=val_interval)
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'lr'
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
