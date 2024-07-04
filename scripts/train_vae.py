from modsiren.datasets_bak import TexturesDataset
from modsiren.vanilla_vae import *
from modsiren.utils import psnr
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from torchvision.utils import make_grid
from skimage import filters
import torch.nn as nn
import torch
import numpy as np
import os
import sys
import yaml
from attrdict import AttrDict
from argparse import ArgumentParser
import matplotlib.pyplot as plt

seed_everything(42)
reconstruction_function = nn.BCELoss()

class FunctionalImagingModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.train_dataset = TexturesDataset(
            root=config.root_dir, filenames=config.train_filenames, num_images=100000, imsize=config.image.size, \
                    channels=config.image.channels, crop_size=config.image.crop_size, random_crop=False)
        # self.val_dataset = TexturesDataset(
            # root=config.root_dir, filenames=config.val_filenames, num_images=100000, imsize=config.image.size, \
                    # channels=config.image.channels, crop_size=config.image.crop_size, random_crop=False)

        # self.model = VAE(3, config.image.size, config.image.size, config.latent.dim)
        self.model = VanillaVAE(3, config.latent.dim)


    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        ground_truth = batch['img']
        model_output = self.forward(batch)

        # Rearrange
        img_out = model_output['model_out']
        # img_out = img_out.permute(0, 2, 1)
        # img_out = img_out.view(ground_truth.shape)
        # img_loss = torch.mean((img_out - ground_truth).pow(2))
        # loss = img_loss

        # BCE = reconstruction_function(img_out, ground_truth.float())
        # BCE = torch.mean((img_out - ground_truth).pow(2))
        mu = model_output['mu']
        logvar = model_output['logvar']

        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        # KLD = torch.sum(KLD_element).mul_(-0.5)

        # loss = BCE + KLD
        loss = self.model.loss_function(img_out, ground_truth.float(), mu, logvar, M_N=config.training.batch_size/len(self.train_dataset))['loss']

        psnr_ = psnr(img_out, ground_truth)

        logs = {'loss': loss,
                'psnr': psnr_}

        if self.global_step % config.logging.scalars_step == 0:
            self.logger.experiment.add_scalar('psnr', psnr_, self.global_step)
            # self.logger.experiment.add_scalar('img_loss', img_loss, self.global_step)

        if self.global_step % config.logging.image_step == 0:
            out_w_gt = torch.cat([img_out[:config.logging.image_vis_count], ground_truth[:config.logging.image_vis_count]], dim=0)
            img_grid = make_grid(out_w_gt, nrow=config.logging.image_vis_count)
            self.logger.experiment.add_image(
                            f'train_w_gt', img_grid.clamp(0, 1), self.global_step)

        return loss

    def configure_optimizers(self):
        train_optimizer = torch.optim.Adam(
            list(self.parameters()), lr=config.training.lr)
        return train_optimizer

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset, batch_size=config.training.batch_size, shuffle=True, pin_memory=True, num_workers=8, drop_last=True)
        return dataloader


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', default='config.yml')
    args = parser.parse_args()
    
    # Load config file
    with open(args.config, 'r') as f:
        try:
            config = AttrDict(yaml.safe_load(f))
        except:
            print('Cant load config')
            exit()

    checkpoint_callback = ModelCheckpoint(
	verbose=True,
        save_top_k=5,
	monitor='psnr',
	mode='max',
        period=10,
    )

    logger = TensorBoardLogger(
        save_dir=config.log_dir,
        name=config.exp_name)

    module = FunctionalImagingModule()
    trainer = Trainer(max_epochs=config.training.epochs,
                      logger=logger,
                      fast_dev_run=False,
                      gpus=[config.gpu],
                      check_val_every_n_epoch=20,
                      profiler=True,
                      checkpoint_callback=checkpoint_callback)
    trainer.fit(module)
