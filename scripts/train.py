#!/bin/env python
"""Train a model."""
import argparse

import pytorch_lightning as pl
import torch as th

from modsiren import callbacks
from modsiren import config
from modsiren import interface


def main():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("config", help="path to a .yml config file.")
    parser.add_argument("--name", help="override name for the experiment.")
    parser.add_argument("--gpu", type=int, help="gpu to run on")
    parser.add_argument(
        "--checkpoint_dir",
        help="directory to save the logs and checkpoints.",
        default="checkpoints")

    args, other_args = parser.parse_known_args()

    # Use GPU by default if available
    if args.gpu is None and th.cuda.is_available():
        args.gpu = 0

    conf = config.get_config(cli_args=other_args, filepath=args.config)

    # Override config if needed
    if args.name is not None:
        conf.name = args.name

    pl.seed_everything(0)

    logger = pl.loggers.TensorBoardLogger(args.checkpoint_dir, name=conf.name)

    cbks = [
        # pl.callbacks.LearningRateMonitor(),
    ]

    if conf.task == "single_instance":
        data = interface.SingleInstanceDataModule(conf)
        model = interface.SingleInstanceTrainingModule(conf, data.num_cells())
        cbks.append(callbacks.SingleImageVisualizationCallback(logger))
        cbks.append(pl.callbacks.ModelCheckpoint(
            monitor='psnr',
            save_top_k=3,
            save_last=True,
            filename="{epoch}-{psnr:.2f}dB", mode="max"))
    else:
        raise NotImplementedError(f"Unknown task {conf.task}")

    trainer = pl.Trainer.from_argparse_args(
        args,
        default_root_dir=args.checkpoint_dir,
        deterministic=True,
        gpus=[args.gpu] if args.gpu is not None else None,
        callbacks=cbks,
        logger=logger,
    )

    trainer.fit(model, data)


if __name__ == "__main__":
    main()
