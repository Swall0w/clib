# import chainer
import glob
import os
import random

import chainer
import numpy as np
import skimage
from skimage import io
from skimage.transform import resize as imresize


class YoloPreprocessedDataset(chainer.dataset.DatasetMixin):
    def __init__(self, dirs=('JPEGImages/', 'labels/'), root='data/',
                 resize=224, tags='voc.names', random=True):
        self.root = root
        self.dirs = dirs
        self.image_dir = self.root + self.dirs[0]
        self.label_dir = self.root + self.dirs[1]
        self.paths = [os.path.split(f)[-1] for f in glob.glob(
            self.image_dir + '*') if ('png'in f or 'jpg' in f)]
        self.tags_path = self.root + tags
        if isinstance(resize, int):
            resize = [resize]
        self.crop_size = resize
        self.random = random
        self.init_tag()

    def __len__(self):
        return len(self.paths)

    def init_tag(self):
        with open(self.tags_path, 'r') as f:
            self.tags = [item.strip() for item in f.readlines()]

    def convert2tuple(self, line):
        line = line.strip().split()
        line[0] = int(line[0])
        line[1] = float(line[1])
        line[2] = float(line[2])
        line[3] = float(line[3])
        line[4] = float(line[4])
        return tuple(line)

    def label_reader(self, labelfile):
        with open(labelfile, 'r') as f:
            lines = [self.convert2tuple(line) for line in f.readlines()]
        ground_truths = []
        for item in lines:
            one_hot_label = np.zeros(len(self.tags))
            one_hot_label[item[0]] = 1
            ground_truths.append({
                'x': item[1],
                'y': item[2],
                'w': item[3],
                'h': item[4],
                'label': item[0],
                'one_hot_label': one_hot_label
            })
        return ground_truths

    def get_example(self, i):
        resize = int(*random.sample(self.crop_size, 1))
        crop_size = (resize, resize)
        imagefile = self.image_dir + self.paths[i]
        labelfile = self.label_dir + self.paths[i].split('.')[0] + '.txt'
        image = io.imread(imagefile)
        labels = self.label_reader(labelfile)

        image = imresize(image, crop_size, mode='reflect')
        image = skimage.img_as_float(image)
        image = image.transpose(2, 0, 1)
        return image, labels
