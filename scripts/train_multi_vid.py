from argparse import ArgumentParser

import numpy as np
import torch
import yaml
from attrdict import AttrDict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from modsiren.datasets_bak import Video, MultiVideoDataset
from modsiren.models_bak import LocalMLP, GlobalMLP
from modsiren.utils import psnr, video_unblockshaped_wc

import matplotlib.pyplot as plt

seed_everything(42)

class FunctionalVideoModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.vid_dataset = MultiVideoDataset(config.vid_path, config.video.channels, config.video.cell_size) 

        if config.model.local:
            self.model = LocalMLP(in_features=3, out_features=config.video.channels, \
                                    hidden_layers=config.model.layers, hidden_features=config.model.width, \
                                    latent_dim = config.latent.dim, \
                                    synthesis_activation=config.model.synthesis_activation, \
                                    modulation_activation=config.model.modulation_activation, \
                                    embedding=config.model.embedding, \
                                    N_freqs=config.model.N_freqs, \
                                    encoder = config.model.encoder, \
                                    patch_res = config.video.cell_size)
        else:
            self.model = GlobalMLP(in_features=3, out_features=config.video.channels, \
                                    hidden_layers=config.model.layers, hidden_features=config.model.width, \
                                    synthesis_activation=config.model.synthesis_activation, \
                                    embedding=config.model.embedding, \
                                    N_freqs=config.model.N_freqs, \
                                    freq_scale=config.model.freq_scale) 


        self.hparams = config.training

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        coords = batch['coords']
        latent_idx = batch['idx']
        ground_truth = batch['img'].view(config.training.batch_size, -1, config.video.channels)

        batch['img'] = batch['img'].permute(0, 4, 1, 2, 3)
        model_output = self.forward(batch)

        output = model_output['model_out'].view(config.training.batch_size, -1, config.video.channels)

        loss = ((output - ground_truth) ** 2).mean()
        psnr_ = psnr(output, ground_truth)

        logs = {'train/loss': loss,
                'train/psnr': psnr_}

        self.log('loss', loss)
        self.log('psnr', psnr_)

        return loss

    def configure_optimizers(self):
        train_optimizer = torch.optim.Adam(
            list(self.parameters()), lr=config.training.lr)
        return train_optimizer

    def train_dataloader(self):
        dataloader = DataLoader(
            self.vid_dataset, shuffle=True, batch_size=self.hparams.batch_size, pin_memory=True, num_workers=8, drop_last=True)
        return dataloader

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', default='config.yml')
    args = parser.parse_args()

    # Load config file
    with open(args.config, 'r') as f:
        config = AttrDict(yaml.safe_load(f))

    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        save_top_k=5,
        monitor='loss',
        mode='min',
        period=5,
    )

    logger = TensorBoardLogger(
        save_dir=config.log_dir,
        name=config.exp_name)

    trainer = Trainer(max_epochs=config.training.epochs,
                      logger=logger,
                      fast_dev_run=False,
                      gpus=[config.gpu],
                      profiler=False,
                      checkpoint_callback=checkpoint_callback,
                      auto_scale_batch_size='power')

    module = FunctionalVideoModule()
    trainer.tune(module)
    trainer.fit(module)
