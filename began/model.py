from __future__ import print_function, division, unicode_literals

from math import log2

import chainer.functions as F
import chainer.links as L
from chainer import Chain


class Generator(Chain):
    def __init__(self, n, out_size, out_channels, embed_size, block_size, device=-1):

        first_xpu_layer = L.Linear(None, n * embed_size * embed_size)
        last_xpu_layer = L.Convolution2D(n, out_channels, 3, stride=1, pad=1)

        if device >= 0:
            first_xpu_layer = first_xpu_layer.to_gpu(device)
            last_xpu_layer = last_xpu_layer.to_gpu(device)

        super(Generator, self).__init__(
            first_layer=first_xpu_layer,
            last_layer=last_xpu_layer)

        self.embed_shape = (n, embed_size, embed_size)
        self.n_blocks = int(log2(out_size / embed_size)) + 1
        self.block_size = block_size

        with self.init_scope():
            for i in range(self.n_blocks * block_size):
                layer = L.Convolution2D(n, n, 3, stride=1, pad=1)
                if device >= 0:
                    layer.to_gpu(device)
                self.__dict__["l{}".format(i)] = layer

    def __call__(self, z):
        h = F.reshape(self.first_layer(z), ((z.shape[0],) + self.embed_shape))

        for i in range(self.n_blocks):
            for j in range(self.block_size):
                h = F.elu(getattr(self, 'l{}'.format(i * j + j))(h))
            if i < self.n_blocks - 1:
                h = F.unpooling_2d(h, ksize=2, stride=2, cover_all=False)
        return self.last_layer(h)


class Discriminator(Chain):
    def __init__(self, n, h, in_size, in_channels, block_size=2, embed_size=8, device=-1):
        super(Discriminator, self).__init__(encoder=Encoder(
            n=n,
            h=h,
            in_size=in_size,
            in_channels=in_channels,
            embed_size=embed_size,
            block_size=block_size,
            device=device
        ), decoder=Decoder(
            n=n,
            out_size=in_size,
            out_channels=in_channels,
            embed_size=embed_size,
            block_size=block_size,
            device=device
        ))

    def __call__(self, x):
        return self.decoder(self.encoder(x))


class Encoder(Chain):
    def __init__(self, n, h, in_size, in_channels, embed_size, block_size, device=-1):

        first_xpu_layer = L.Convolution2D(in_channels, n, 3, stride=1, pad=1)
        last_xpu_layer = L.Linear(None, h)

        if device >= 0:
            first_xpu_layer = first_xpu_layer.to_gpu(device)

        super(Encoder, self).__init__(
            first_layer=first_xpu_layer,
            last_layer=last_xpu_layer
        )

        self.n_blocks = int(log2(in_size / embed_size)) + 1
        self.block_size = block_size

        with self.init_scope():
            for i in range(self.n_blocks):
                n_in = (i + 1) * n
                n_out = (i + 2) * n if i < self.n_blocks - 1 else n_in
                for j in range(block_size - 1):
                    ij_layer = L.Convolution2D(n_in, n_in, 3, stride=1, pad=1)
                    if device >= 0:
                        ij_layer = ij_layer.to_gpu(device)
                    self.__dict__['l{}'.format(i * block_size + j)] = ij_layer
                i_layer = L.Convolution2D(n_in, n_out, 3, stride=1, pad=1)
                if device >= 0:
                    i_layer = i_layer.to_gpu(device)
                self.__dict__['l{}'.format(i * block_size + block_size - 1)] = i_layer

    def __call__(self, x):
        h = F.elu(self.first_layer(x))
        for i in range(self.n_blocks):
            for j in range(self.block_size):
                h = getattr(self, 'l{}'.format(i * self.block_size + j))(h)
                h = F.elu(h)
            if i < self.n_blocks - 1:
                h = F.max_pooling_2d(h, ksize=2, stride=2)
        return self.last_layer(h)


class Decoder(Chain):
    def __init__(self, n, out_size, out_channels, embed_size, block_size, device=-1):

        first_xpu_layer = L.Linear(None, n * embed_size * embed_size)
        last_xpu_layer = L.Convolution2D(n, out_channels, 3, stride=1, pad=1)

        if device >= 0:
            first_xpu_layer = first_xpu_layer.to_gpu(device)
            last_xpu_layer = last_xpu_layer.to_gpu(device)

        super(Decoder, self).__init__(
            first_layer=first_xpu_layer,
            last_layer=last_xpu_layer
        )

        self.embed_shape = (n, embed_size, embed_size)
        self.n_blocks = int(log2(out_size / embed_size)) + 1
        self.block_size = block_size
        with self.init_scope():
            for i in range(self.n_blocks * block_size):
                layer = L.Convolution2D(n, n, 3, stride=1, pad=1)
                if device >= 0:
                    layer = layer.to_gpu(device)
                self.__dict__["l{}".format(i)] = layer

    def __call__(self, x):
        h = F.reshape(self.first_layer(x), ((x.shape[0],) + self.embed_shape))
        for i in range(self.n_blocks):
            for j in range(self.block_size):
                h = F.elu(getattr(self, 'l{}'.format(i * j + j))(h))
            if i < self.n_blocks - 1:
                h = F.unpooling_2d(h, ksize=2, stride=2, cover_all=False)

        return self.last_layer(h)
