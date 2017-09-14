import six
from chainer import Variable, training
from chainer.dataset import iterator as itr_module
from clib.converts import format_image_size
from chainer import cuda
import numpy

def _concat_arrays(arrays):
    if (not isinstance(arrays[0], numpy.ndarray) and
            not isinstance(arrays[0], cuda.ndarray)):
        arrays = numpy.asarray(arrays)
    xp = cuda.get_array_module(arrays[0])
    with cuda.get_device(arrays[0]):
        return xp.concatenate([array[None] for array in arrays])

def _to_device(device, x):
    if device is None:
        return x
    elif device < 0:
        return cuda.to_cpu(x)
    else:
        return cuda.to_gpu(x, device, cuda.Stream.null)


def darknet_converter(batch, device=None):

    if len(batch) == 0:
        raise ValueError('batch is empty')

    first_elem = batch[0]
    if isinstance(first_elem, tuple):
        x = _to_device(device, _concat_arrays([item[0] for item in batch]))
        x = Variable(x)
        label = [item[1][0]['one_hot_label'] for item in batch]
        label = Variable(numpy.array(label, dtype=numpy.float32))
        if device >= 0:
            label.to_gpu()
        return x, label


def yolo_converter(batch, device=None):

    if len(batch) == 0:
        raise ValueError('batch is empty')

    batch = format_image_size.batch(batch)

    first_elem = batch[0]
    if isinstance(first_elem, tuple):
        x = _to_device(device, _concat_arrays([item[0] for item in batch]))
        x = Variable(x)
        label = [item[1] for item in batch]
        return x, label
