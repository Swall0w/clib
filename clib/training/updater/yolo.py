import numpy as np
import six
from chainer import Variable, training
from chainer.dataset import iterator as itr_module
from clib.converts import format_image_size


def darknet_converter(batch, device=None):
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

    if len(batch) == 0:
        raise ValueError('batch is empty')

    first_elem = batch[0]
    if isinstance(first_elem, tuple):
        x = _to_device(device, _concat_arrays([item[0] for item in batch]))
        x = Variable(x)
        label = [item[1][0]['one_hot_label'] for item in batch]
        label = Variable(np.array(label, dtype=np.float32))
        if device >= 0:
            label.to_gpu()
        return x, label


def yolo_converter(batch, device=None):
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

    if len(batch) == 0:
        raise ValueError('batch is empty')

    first_elem = batch[0]
    if isinstance(first_elem, tuple):
        x = _to_device(device, _concat_arrays([item[0] for item in batch]))
        x = Variable(x)
        label = [item[1] for item in batch]
        return x, label


class YoloUpdater(training.StandardUpdater):
    def __init__(self, iterator, optimizer, converter=yolo_converter,
                 device=None, loss_func=None):
        if isinstance(iterator, itr_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator

        if not isinstance(optimizer, dict):
            optimizer = {'main': optimizer}
        self._optimizers = optimizer

        if device is not None and device >= 0:
            for optimizer in six.itervalues(self._optimizers):
                optimizer.target.to_gpu(device)

        self.converter = converter
        self.loss_func = loss_func
        self.device = device
        self.iteration = 0

    def update_core(self):
        batch = self._iterators['main'].next()
        batch = format_image_size.batch(batch)
        in_arrays = self.converter(batch, self.device)

        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target

        if isinstance(in_arrays, tuple):
            optimizer.update(loss_func, *in_arrays)
        elif isinstance(in_arrays, dict):
            optimizer.update(loss_func, **in_arrays)
        else:
            optimizer.update(loss_func, in_arrays)

    def converter(self, batch, device=None):
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

        if len(batch) == 0:
            raise ValueError('batch is empty')

        first_elem = batch[0]
        if isinstance(first_elem, tuple):
            x = _to_device(device, _concat_arrays([item[0] for item in batch]))
            x = Variable(x)
            label = [item[1] for item in batch]
            return x, label
