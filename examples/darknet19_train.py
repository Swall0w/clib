import argparse
import numpy as np
import chainer
from chainer import optimizers, training, iterators, Variable
from chainer.training import extensions
from chainer.dataset import iterator as itr_module

from yolo.model.darknet import Darknet19, Darknet19Predictor
from yolo.trainer.dataset import PreprocessedDataset
from yolo.trainer.update import darknet_converter

def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch','-b',type=int,default=100,help='number of minibatch')
    parser.add_argument('--epoch','-e',type=int,default=100,help='number of epoch')
    parser.add_argument('--gpu','-g',type=int,default=-1,help='number of gpu')
    parser.add_argument('--output','-o',default='result',help='output directory')
    parser.add_argument('--resume','-r',default='',help='resume the training from snapshot')
    parser.add_argument('--unit','-u',type=int,default=500,help='number of unit')
    parser.add_argument('--lr',default=0.001,help='learning rate')
    parser.add_argument('--momentum',default=0.9,help='MomentumSGD')
    parser.add_argument('--weight_decay','-wd',default=0.0005,help='WeightDecay')
    parser.add_argument('--resize','-s',type=int,default=224,help='number of unit')
    return parser.parse_args()

def main():
    args = arg()
    model = Darknet19Predictor(Darknet19())
    if args.resize == 448:
        chainer.serializers.load_npz(args.output+'/darknet19.weights',model)
        model.predictor.finetune = True
        print('darknet19 model loaded.')
    chainer.config.train = True
#    model.predictor.train = True


    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    optimizer = optimizers.MomentumSGD(lr=args.lr, momentum=args.momentum)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

    print('resolution: ',args.resize)
    train = PreprocessedDataset(resize=args.resize)
    train_itr = iterators.SerialIterator(dataset=train,batch_size=args.batch)
    updater = training.StandardUpdater(train_itr, optimizer,converter=darknet_converter,device=args.gpu)
    trainer = training.Trainer(updater,(args.epoch,'epoch'),out=args.output)

    trainer.extend(extensions.ExponentialShift('lr',0.97),trigger=(1,'epoch'))
    trainer.extend(extensions.observe_lr())

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch','iteration','main/loss','main/accuracy','lr']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()
    if args.resize == 448:
        chainer.serializers.save_npz(args.output+'/darknet19_448.weights',model.to_cpu())
    else:
        chainer.serializers.save_npz(args.output+'/darknet19.weights',model.to_cpu())
    print('end')

if __name__ == '__main__':
    main()

