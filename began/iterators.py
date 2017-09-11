import cupy
import numpy
from chainer.dataset import iterator


def to_tuple(x):
    if hasattr(x, '__getitem__'):
        return x
    return x,


class UniformNoiseGenerator(object):
    def __init__(self, low, high, size, device=-1):

        if device >= 0:
            self.xp = cupy
        else:
            self.xp = numpy

        self.low = low
        self.high = high
        self.size = to_tuple(size)

    def __call__(self, batch_size):
        return self.xp.random.uniform(self.low, self.high, (batch_size,) + self.size).astype(numpy.float32)


class RandomNoiseIterator(iterator.Iterator):
    def __init__(self, noise_generator, batch_size):
        self.noise_generator = noise_generator
        self.batch_size = batch_size

    def __next__(self):
        batch = self.noise_generator(self.batch_size)
        return batch
