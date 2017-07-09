#!/usr/bin/env python

from chainer import serializers
import argparse
from clib.links.model.yolo.darknet import Darknet19Predictor, Darknet19
from clib.links.model.yolo.yolov2 import YOLOv2, YOLOv2Predictor


def copy_conv_layer(src, dst, layers):
    for i in layers:
        src_layer = eval("src.conv%d" % i)
        dst_layer = eval("dst.conv%d" % i)
        dst_layer.W = src_layer.W
        dst_layer.b = src_layer.b


def copy_bias_layer(src, dst, layers):
    for i in layers:
        src_layer = eval("src.bias%d" % i)
        dst_layer = eval("dst.bias%d" % i)
        dst_layer.b = src_layer.b


def copy_bn_layer(src, dst, layers):
    for i in layers:
        src_layer = eval("src.bn%d" % i)
        dst_layer = eval("dst.bn%d" % i)
        dst_layer.N = src_layer.N
        dst_layer.avg_var = src_layer.avg_var
        dst_layer.avg_mean = src_layer.avg_mean
        dst_layer.gamma = src_layer.gamma
        dst_layer.eps = src_layer.eps


def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls', '-c', type=int, default=20,
                        help='number of classes')
    parser.add_argument('--box', '-b', type=int, default=5,
                        help='number of boxes')
    parser.add_argument('--layer', '-l', type=int, default=18,
                        help='number of partial layer')
    parser.add_argument('--input', '-i', type=str,
                        default='result/darknet19_448.weights',
                        help='number of partial layer')
    parser.add_argument('--output', '-o', type=str,
                        default='result/darknet19_448.conv.23',
                        help='number of partial layer')
    return parser.parse_args()


def main():
    args = arg()
    print("loading original model...")
    model = Darknet19Predictor(Darknet19())
    serializers.load_npz(args.input, model)  # load saved model

    yolov2 = YOLOv2(n_classes=args.cls, n_boxes=args.box)
    copy_conv_layer(model.predictor, yolov2, range(1, args.layer+1))
    copy_bias_layer(model.predictor, yolov2, range(1, args.layer+1))
    copy_bn_layer(model.predictor, yolov2, range(1, args.layer+1))
    model = YOLOv2Predictor(yolov2)

    print("saving model to %s" % (args.output))
    serializers.save_npz("%s" % (args.output), model)


if __name__ == '__main__':
    main()
