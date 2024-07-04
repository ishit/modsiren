from modsiren.utils import *
from modsiren.datasets_bak import DiligentDataset
from modsiren.models_bak import LocalMLP, GlobalMLP
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from skimage import filters
import torch.nn as nn
import torch
import numpy as np
import os
import sys
import yaml
from attrdict import AttrDict
from argparse import ArgumentParser

seed_everything(42)


class FunctionalImagingModule(LightningModule):
    def __init__(self):
        super().__init__()
        patch_res = config.image.size // config.image.cells

        self.train_dataset = DiligentDataset(imsize=config.image.size)
        # Function Signature
        # def __init__(self, in_features, out_features, hidden_layers=4, hidden_features=256, latent_dim=None, 
                    # synthesis_activation=None, modulation_activation=None, embedding=None, N_freqs=None, 
                    # encoder=False, patch_res=patch_res):
        self.model = LocalMLP(in_features=2, out_features=config.image.channels, \
                                hidden_layers=config.model.layers, hidden_features=config.model.width, \
                                latent_dim = config.latent.dim, \
                                synthesis_activation=config.model.synthesis_activation, \
                                modulation_activation=config.model.modulation_activation, \
                                embedding=config.model.embedding, \
                                N_freqs=config.model.N_freqs, \
                                encoder = config.model.encoder, \
                                patch_res = (patch_res, patch_res))

    def forward(self, batch):
        BS, PS, D = batch['coords'].shape
        ground_truth = batch['img']
        idx = batch['idx']

        return self.model(batch)

    def training_step(self, batch, batch_idx):
        ground_truth = batch['img']
        model_output = self.forward(batch)

        # Rearrange
        img_out = model_output['model_out']
        img_out = img_out.permute(0, 2, 1) # BS, C, PS
        img_out = img_out.view(
            config.training.batch_size, config.image.channels, config.image.imsize, config.image.imsize)


        img_loss = torch.mean(((img_out - ground_truth)).pow(2))
        loss = img_loss

        psnr_ = psnr(img_out, ground_truth)

        logs = {'loss': loss,
                'l2': img_loss,
                'psnr': psnr_}

        if self.global_step % config.logging.image_step == 0:
            self.log_images()
            self.log_z_hist()

        if self.global_step % config.logging.scalars_step == 0:
            self.logger.experiment.add_scalar('psnr', psnr_, self.global_step)
            self.logger.experiment.add_scalar('img_loss', img_loss, self.global_step)

        return loss

    def configure_optimizers(self):
        train_optimizer = torch.optim.Adam(
            list(self.parameters()), lr=config.training.lr)
        return train_optimizer

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset, batch_size=config.training.batch_size, shuffle=True, pin_memory=True, num_workers=8)
        return dataloader

    def log_z_hist(self):
        self.logger.experiment.add_histogram('z_coords', self.train_codes, self.global_step)

    def log_images(self):
        out_im = []
        gt_im = []
        dataloader = DataLoader(
            self.train_dataset, shuffle=False, batch_size=config.training.batch_size)

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                ground_truth = batch['img'].cuda()
                batch['coords'] = batch['coords'].cuda()
                if not config.model.local:
                    batch['global_coords'] = batch['global_coords'].cuda()
                model_output = self.forward(batch)

                # Rearrange
                img_out = model_output['model_out']
                img_out = img_out.permute(0, 2, 1) # BS, C, PS
                img_out = img_out.view(
                    config.training.batch_size, config.image.channels, self.train_dataset.cell_width, self.train_dataset.cell_width)

                out_im.append(img_out)
                gt_im.append(ground_truth)

                if len(out_im) * config.training.batch_size >= self.train_dataset.n_patches:
                    break

        out_im = torch.cat(out_im, dim=0)
        gt_im = torch.cat(gt_im, dim=0)

        out_channels = []
        gt_channels = []


        if config.model.local:
            for i in range(config.image.channels):
                out_i = unblockshaped_tensor_single(
                        out_im[:, i], 2*config.image.size, 2*config.image.size)
                gt_i = unblockshaped_tensor_single(
                        gt_im[:, i], 2*config.image.size, 2*config.image.size)

                out_i = self.linear_blending(out_i)
                out_i = self.linear_blending(out_i.permute(1, 0)).permute(1, 0)
                gt_i = self.linear_blending(gt_i)
                gt_i = self.linear_blending(gt_i.permute(1, 0)).permute(1, 0)

                out_channels.append(out_i)
                gt_channels.append(gt_i)

        else:
            for i in range(config.image.channels):
                out_i = unblockshaped_tensor_single(
                        out_im[:, i], config.image.size, config.image.size)
                gt_i = unblockshaped_tensor_single(
                        gt_im[:, i], config.image.size, config.image.size)

                out_channels.append(out_i)
                gt_channels.append(gt_i)

        out_im = torch.stack(out_channels)
        gt_im = torch.stack(gt_channels)
        self.logger.experiment.add_image(
                        'est', out_im.clamp(0, 1), self.global_step)
        psnr_ = psnr(out_im, gt_im)
        # self.logger.experiment.add_scalar('val_psnr', psnr_, self.global_step)
        self.log('val_psnr', psnr_)

        if self.global_step < 2:
            self.logger.experiment.add_image(
                'gt', gt_im.clamp(0, 1), self.global_step)


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
        save_top_k=2,
	monitor='val_psnr',
	mode='max',
        period=10,
    )

    logger = TensorBoardLogger(
        save_dir=config.log_dir,
        name=config.exp_name)

    trainer = Trainer(max_epochs=config.training.epochs,
                      logger=logger,
                      fast_dev_run=False,
                      gpus=[config.gpu],
                      profiler=True,
                      checkpoint_callback=checkpoint_callback)

    module = FunctionalImagingModule()
    trainer.fit(module)

