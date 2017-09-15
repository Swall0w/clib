import glob
import os
import random

import chainer
import numpy as np
import skimage
from skimage import io
from skimage.transform import resize as imresize
from .basedataset import BaseLabeledImageDataset


class YoloPreprocessedDataset(BaseLabeledImageDataset):
    def get_example(self, i):
        if isinstance(self.resize, int):
            self.crop_size = [self.resize]
        else:
            self.crop_size = self.resize

        resize = int(*random.sample(self.crop_size, 1))
        crop_size = (resize, resize)
        full_path, label = self._pairs[i]
        image = io.imread(full_path)
        labels = self.label_reader(label)
        image = imresize(image, crop_size, mode='reflect')
        image = skimage.img_as_float(image)
        return image.transpose(2, 0, 1), labels

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
            one_hot_label = np.zeros(len(self.label_dict))
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
