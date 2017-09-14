import argparse

import chainer
from chainer import optimizers, training
from chainer.training import extensions
from clib.links.model.yolo.yolov2 import YOLOv2, YOLOv2Predictor
from clib.training.dataset import YoloPreprocessedDataset
from clib.training.updater import yolo_converter


def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', '-b', type=int, default=100,
                        help='number of minibatch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='number of epoch')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='number of gpu')
    parser.add_argument('--output', '-o', default='result',
                        help='output directory')
    parser.add_argument('--model', '-m', default='darknet19_448.conv.23',
                        help='Initial model')
    parser.add_argument('--resume', '-r', default='',
                        help='resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=500,
                        help='number of unit')
    parser.add_argument('--lr', default=1e-5, type=float,
                        help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='MomentumSGD')
    parser.add_argument('--weight_decay', '-wd', default=0.0005,
                        type=float, help='WeightDecay')
    parser.add_argument('--cls', '-c', type=int, default=20,
                        help='number of classes')
    parser.add_argument('--box', type=int, default=5,
                        help='number of boxes')
    parser.add_argument('--images', default='JPEGImages/',
                        help='Directory name that contains images')
    parser.add_argument('--labels', default='labels/',
                        help='Directory name that contains labels')
    parser.add_argument('--tags', default='voc.names',
                        help='label list')
    return parser.parse_args()


def main():
    args = arg()
    resize = [320, 352, 384, 416, 448]

    model = YOLOv2Predictor(YOLOv2(n_classes=args.cls, n_boxes=args.box))
    chainer.serializers.load_npz(args.output + '/' + args.model, model)
    model.predictor.finetune = False
    model.predictor.train = True

    chainer.config.train = True
    print('{0} model loaded.'.format(args.model))

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    optimizer = optimizers.MomentumSGD(lr=args.lr, momentum=args.momentum)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

    train = YoloPreprocessedDataset(
        dirs=(args.images, args.labels), resize=resize, tags=args.tags)
    train_itr = chainer.iterators.SerialIterator(
        dataset=train, batch_size=args.batch)
    updater = training.StandardUpdater(
        train_itr, optimizer, converter=yolo_converter, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.output)

    trainer.extend(
        extensions.ExponentialShift('lr', 0.97), trigger=(1, 'epoch'))
    trainer.extend(extensions.observe_lr())

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'elapsed_time', 'iteration', 'main/loss', 'lr',
         'main/x_loss', 'main/y_loss', 'main/w_loss', 'main/h_loss',
         'main/c_loss', 'main/p_loss']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()
    chainer.serializers.save_npz(args.output+'/yolo.weights', model.to_cpu())
    print('end')


if __name__ == '__main__':
    main()
