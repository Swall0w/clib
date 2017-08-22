import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from clib.links.model.recognition import Alex
from clib.training.dataset import UnifiedLabeledImageDataset
from PIL import Image


def arg():
    parser = argparse.ArgumentParser(description='VGG Finetune')
    parser.add_argument('--train', help='Path to training image-label list file')
    parser.add_argument('--test', help='Path to validation image-label list file')
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
    parser.add_argument('--test_batchsize', '-b', type=int, default=250,
                        help='Validation minibatch size')
    parser.add_argument('--n_class', '-c', type=int,
                        help='Number of image class')
    parser.add_argument('--resize', type=int, default=256,
                        help='Number of image class')
    return parser.parse_args()


def main():
    args = arg()
    print('train dataset : ', args.train)

    model = L.Classifier(Alex(args.n_class))
    if args.initmodel:
        print('Load model from ', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    train = UnifiedLabeledImageDataset(pairs=args.train,
                                       resize=(args.resize, args.resize))
    test = UnifiedLabeledImageDataset(pairs=args.test,
                                       resize=(args.resize, args.resize))
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()
    chainer.serializers.save_npz('result.weights', model.to_cpu())

if __name__ == '__main__':
    main()
