import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from clib.training.dataset import UnifiedLabeledImageDataset
from clib.utils import arg_recognition
from PIL import Image


class VGGFineTune(chainer.Chain):
    def __init__(self, n_image):
        super(VGGFineTune, self).__init__()
        with self.init_scope():
            self.vgg = L.VGG16Layers()
            self.fc8 = L.Linear(4096, 1000)
            self.fc9 = L.Linear(None, n_image)

    def __call__(self, x):
        h = self.vgg(x, layers=['fc7'])['fc7']
        h = F.dropout(F.relu(self.fc8(h)))
        h = self.fc9(h)
        return h


def main():
    args = arg_recognition()

    model = L.Classifier(VGGFineTune(args.n_class))
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
