hparams = {
        "exp_name": "SimpleEncoderModSirenlarge64",
        "latent_dim": 64,
        "cells": 8,
        "net_width": 128,
        "num_images": 100000,
        "filter_radius": 0,
        "n_layers": 3,
        "imsize": 128,
        "channels": 3,
        "train_datapath": "img_align_celeba",
        "val_datapath": "img_align_celeba/val",
        "epochs": 100000,
        "batch_size": 8,
        "lr": 5e-5,
        "val_lr": 1e-2,
        "val_optim_steps": 1,
        "val_frequency": 10000,
        "l2_weight": 1,
        "reg_weight": 1e-4,
        "encoder": True,
        "hyper": False,
        "shuffle": True,
        "gpus": [7]
        }

hparams['patch_res'] = hparams['imsize'] // hparams['cells']
