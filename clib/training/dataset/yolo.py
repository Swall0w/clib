import glob
import os.path
import random

import chainer
import numpy as np
import skimage
from skimage import io
from skimage.transform import resize as imresize
from .basedataset import BaseLabeledImageDataset
from clib.datasets import voc_load
from clib.utils import get_index_from_label


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
        labels = label_reader(label, self.label_dict)
        image = imresize(image, crop_size, mode='reflect')
        image = skimage.img_as_float(image)
        return image.transpose(2, 0, 1), labels

def convert2tuple(line):
    line = line.strip().split()
    line[0] = int(line[0])
    line[1] = float(line[1])
    line[2] = float(line[2])
    line[3] = float(line[3])
    line[4] = float(line[4])
    return tuple(line)

def label_reader(labelfile, label_dict):
    root, ext = os.path.splitext(labelfile)
    if ext == '.txt':
        with open(labelfile, 'r') as f:
            lines = [convert2tuple(line) for line in f.readlines()]
    elif ext == '.xml':
        size, label = voc_load(labelfile)
        lines = xml_to_yolo(size, label, label_dict)
    else:
        pass

    ground_truths = []
    for item in lines:
        one_hot_label = np.zeros(len(label_dict))
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

def xml_to_yolo(size, label, label_dict):
    rets = []
    for item in label:
        ret = []
        b = (float(item['xmin']), float(item['ymin']),
             float(item['xmax']), float(item['ymax']))
        bb = convert_coordinate_to_relative(size, b)
        ret.append(get_index_from_label(label_dict, item['label']))
        ret.extend(list(bb))
        rets.append(tuple(ret))
    return rets

def convert_coordinate_to_relative(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)

