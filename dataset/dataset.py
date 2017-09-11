import glob

import chainer
import numpy as np
from PIL import Image


class CelebaDataSet(chainer.dataset.DatasetMixin):
    def __init__(self, size=None, crop=None):
        if isinstance(size, int):
            size = size, size

        if crop == 'face':
            w, h = 128, 128
            upper_left, bottom_right = (25, 50), (25 + w, 50 + h)
            crop = upper_left + bottom_right

        paths = glob.glob('dataset/img/celeba/*.jpg')

        self.paths = paths
        self.size = size
        self.crop = crop

    def __len__(self):
        return len(self.paths)

    def get_example(self, i):
        path = self.paths[i]

        with Image.open(path) as f:
            if self.crop is not None:
                f = f.crop(self.crop)
            if self.size is not None:
                f = f.resize(self.size, Image.ANTIALIAS)
            im = np.asarray(f, dtype=np.float32)

        im = im.transpose((2, 0, 1))
        im *= (2 / 255)
        im -= 1
        return im


if __name__ == '__main__':
    dataset = CelebaDataSet()
    example = dataset.get_example(1)
