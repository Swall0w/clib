import argparse


def arg_recognition():
    parser = argparse.ArgumentParser(description='Object Recognition')
    parser.add_argument('--train',
                        help='Path to training image-label list file')
    parser.add_argument('--test',
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
    parser.add_argument('--test_batchsize', '-b', type=int, default=250,
                        help='Validation minibatch size')
    parser.add_argument('--n_class', '-c', type=int,
                        help='Number of image class')
    parser.add_argument('--resize', type=int, default=256,
                        help='Number of image class')

    return parser.parse_args()
