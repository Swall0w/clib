#import chainer
import numpy as np
import glob
import cv2
import os
import random

class DatasetMixin(object):
    """Default implementation of dataset indexing.
    DatasetMixin provides the :meth:`__getitem__` operator. The default
    implementation uses :meth:`get_example` to extract each example, and
    combines the results into a list. This mixin makes it easy to implement a
    new dataset that does not support efficient slicing.

    Dataset implementation using DatasetMixin still has to provide the
    :meth:`__len__` operator explicitly.

    """

    def __getitem__(self, index):
        """Returns an example or a sequence of examples.

        It implements the standard Python indexing. It uses the
        :meth:`get_example` method by default, but it may be overridden by the
        implementation to, for example, improve the slicing performance.

        """
        if isinstance(index, slice):
            current, stop, step = index.indices(len(self))
            ret = []
            while current < stop and step > 0 or current > stop and step < 0:
                ret.append(self.get_example(current))
                current += step
            return ret
        else:
            return self.get_example(index)

    def __len__(self):
        """Returns the number of data points."""
        raise NotImplementedError

    def get_example(self, i):
        """Returns the i-th example.  
        Implementations should override it. It should raise :class:`IndexError`
        """
        raise NotImplementedError

#class PreprocessedDataset(chainer.dataset.DatasetMixin):
class PreprocessedDataset(DatasetMixin):
    def __init__(self, dirs=('JPEGImages/','labels/'), root='data/', resize=224, tags = 'voc.names',random=True):
        self.root = root
        self.dirs = dirs
        self.image_dir = self.root + self.dirs[0]
        self.label_dir = self.root + self.dirs[1]
        self.paths = [os.path.split(f)[-1] for f in glob.glob(self.image_dir + '*') if ('png'in f or 'jpg' in f)]
        self.tags_path = self.root + tags
        if isinstance(resize, int):
            resize = [resize]
        self.crop_size = resize
        self.random = random
        self.init_tag()
    def __len__(self):
        return len(self.paths)
    def init_tag(self):
        with open(self.tags_path,'r') as f:
            self.tags = [item.strip() for item in f.readlines()]

    def convert2tuple(self,line):
        line = line.strip().split()
        line[0] = int(line[0])
        line[1] = float(line[1])
        line[2] = float(line[2])
        line[3] = float(line[3])
        line[4] = float(line[4])
        return tuple(line)
        
    def label_reader(self,labelfile):
        with open(labelfile,'r') as f:
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

    def get_example(self,i):
        resize = int(*random.sample(self.crop_size,1))
        crop_size = (resize,resize)
#        print(self.paths[i])
        imagefile = self.image_dir + self.paths[i]
        labelfile = self.label_dir + self.paths[i].split('.')[0] + '.txt'
        image = cv2.imread(imagefile, cv2.IMREAD_UNCHANGED)
        labels = self.label_reader(labelfile)

        image = cv2.resize(image,crop_size)
        image = image[:,:,:3]
        image = np.asarray(image,dtype=np.float32)
        image *= (1.0 / 255.0)  # Scale to [0, 1]
        image = image.transpose(2, 0, 1)
        return image, labels
