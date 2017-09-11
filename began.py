from chainer import iterators, optimizers, training
from chainer.training import extensions

import config
import output
from began.iterators import RandomNoiseIterator, UniformNoiseGenerator
from began.model import Generator, Discriminator
from began.updater import BEGANUpdater
from dataset.dataset import CelebaDataSet

if __name__ == '__main__':
    args = config.parse_args()

    train = CelebaDataSet(size=args.celeba_scale, crop='face')
    train_iter = iterators.SerialIterator(train, args.batch_size)

    z_iter = RandomNoiseIterator(UniformNoiseGenerator(-1, 1, args.n_z, args.gpu), args.batch_size)

    g = Generator(
        n=args.g_n,
        out_size=train[0].shape[1],
        out_channels=train[0].shape[0],
        block_size=args.g_block_size,
        embed_size=args.g_embed_size,
        device=args.gpu
    )

    d = Discriminator(
        n=args.d_n,
        h=args.n_h,
        in_size=train[0].shape[1],
        in_channels=train[0].shape[0],
        block_size=args.d_block_size,
        embed_size=args.d_embed_size,
        device=args.gpu
    )

    optimizer_generator = optimizers.Adam(alpha=args.g_lr, beta1=0.5)
    optimizer_discriminator = optimizers.Adam(alpha=args.d_lr, beta1=0.5)

    optimizer_generator.setup(g)
    optimizer_discriminator.setup(d)

    updater = BEGANUpdater(
        iterator=train_iter,
        noise_iterator=z_iter,
        optimizer_generator=optimizer_generator,
        optimizer_discriminator=optimizer_discriminator,
        generator_lr_decay_interval=args.g_lr_decay_interval,
        discriminator_lr_interval=args.g_lr_decay_interval,
        gamma=args.gamma,
        k_0=args.k_0,
        lambda_k=args.lambda_k,
        loss_norm=args.loss_norm,
        device=args.gpu)

    trainer = training.Trainer(updater, out=args.out_dir, stop_trigger=(args.iterations, 'iteration'))
    trainer.extend(extensions.LogReport(trigger=(args.report_interval, 'iteration')))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.PrintReport(['iteration',
                                           'convergence',
                                           'gen/loss',
                                           'dis/loss',
                                           'k']))
    trainer.extend(output.OutputGeneratedData(), trigger=(args.report_interval, 'iteration'))
    trainer.run()
