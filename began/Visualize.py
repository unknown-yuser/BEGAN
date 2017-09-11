from __future__ import print_function, division

import math
import sys

import numpy
from PIL import Image


def save_image(path, image):
    n, c, w, h = image.shape

    rows = math.ceil(math.sqrt(n))
    cols = rows if n % rows == 0 else rows - 1

    if sys.version_info[0] == 2:
        rows = int(rows)
        cols = int(cols)

    if c == 3:
        image = image.reshape((rows, cols, 3, h, w))
        image = image.transpose(0, 3, 1, 4, 2)
        image = image.reshape((rows * h, cols * w, 3))
    else:
        image = image.reshape((rows, cols, 1, h, w))
        image = image.transpose(0, 3, 1, 4, 2)
        image = image.reshape(rows * h, cols * w)

    image = image.astype(numpy.uint8)
    Image.fromarray(image).save(path)
