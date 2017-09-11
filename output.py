import os

import numpy
from chainer import training, cuda

from began import Visualize


class OutputGeneratedData(training.extension.Extension):
    def __init__(self, dirname='output', extension='png'):
        self.dirname = dirname
        self.extension = extension

    def __call__(self, trainer):
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname, exist_ok=True)

        x = self.generate_data(trainer)
        x = numpy.clip((x + 1.0) / 2.0 * 255.0, 0.0, 255.0)

        filename = 'g_{}.{}'.format(trainer.updater.iteration, self.extension)
        filename = os.path.join(self.dirname, filename)
        Visualize.save_image(filename, x)

    def generate_data(self, trainer):
        x = trainer.updater.sample()
        x = x.data
        if cuda.available and isinstance(x, cuda.ndarray):
            x = cuda.to_cpu(x)
        return x
