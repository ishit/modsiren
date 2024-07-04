from modsiren.utils import *
import modsiren.datasets_bak as dataset
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
        self.train_dataset = dataset.celebAGeneralization(
            filenames=config.train_filenames, num_images=1024, imsize=config.image.size, \
                    patch_res=patch_res, channels=config.image.channels, mode='train')
        self.val_dataset = dataset.celebAGeneralization(
            filenames=config.val_filenames, num_images=1024, imsize=config.image.size, \
                    patch_res=patch_res, channels=config.image.channels, mode='val')

        ## Function Signature
        # def __init__(self, in_features, out_features, hidden_layers=4, hidden_features=256, latent_dim=None, 
                    # synthesis_activation=None, modulation_activation=None, embedding=None, N_freqs=None, 
                    # encoder=False, patch_res=patch_res):
        if config.model.local:
            self.model = LocalMLP(in_features=2, out_features=config.image.channels, \
                                    hidden_layers=config.model.layers, hidden_features=config.model.width, \
                                    latent_dim = config.latent.dim, \
                                    synthesis_activation=config.model.synthesis_activation, \
                                    modulation_activation=config.model.modulation_activation, \
                                    embedding=config.model.embedding, \
                                    N_freqs=config.model.N_freqs, \
                                    encoder = config.model.encoder, \
                                    patch_res = (patch_res, patch_res)) 
        else:
            self.model = GlobalMLP(in_features=2, out_features=config.image.channels, \
                                    hidden_layers=config.model.layers, hidden_features=config.model.width, \
                                    synthesis_activation=config.model.synthesis_activation, \
                                    embedding=config.model.embedding, \
                                    N_freqs=config.model.N_freqs) 

    def forward(self, batch):
        BS, PS, D = batch['coords'].shape
        ground_truth = batch['img']
        idx = batch['idx']

        return self.model(batch)

    def linear_blending(self, im):
        cell_width = self.train_dataset.cell_width 
        idcs = np.arange(cell_width // 2, 2 * config.image.size -
                         cell_width, cell_width)
        all_idcs = []
        for i in range(cell_width // 2):
            all_idcs.append(idcs + i)
        all_idcs = np.sort(np.concatenate(all_idcs))

        xx, _ = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
        xx = xx % (cell_width // 2)
        xx = xx / (cell_width // 2 - 1)
        xx = (1 - xx)
        xx = torch.FloatTensor(xx).to(im.device)

        final_im_1 = im[:, all_idcs] * xx[:, all_idcs]
        final_im_2 = im[:, all_idcs+cell_width//2] * \
            (1 - xx[:, all_idcs + cell_width // 2])
        final_im = final_im_1 + final_im_2

        final_im = torch.cat([im[:, cell_width // 4:cell_width // 2],
                                final_im, im[:, -cell_width // 2:-cell_width // 3]], dim=1)
        return final_im

    def training_step(self, batch, batch_idx):
        ground_truth = batch['img']
        model_output = self.forward(batch)

        # Rearrange
        img_out = model_output['model_out']
        img_out = img_out.permute(0, 2, 1) # BS, C, PS
        img_out = img_out.view(
            config.training.batch_size, config.image.channels, self.train_dataset.cell_width, self.train_dataset.cell_width)

        img_loss = torch.mean((img_out - ground_truth).pow(2))
        loss = img_loss

        psnr_ = psnr(img_out, ground_truth)

        logs = {'loss': loss,
                'l2': img_loss,
                'psnr': psnr_}

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
            self.train_dataset, batch_size=config.training.batch_size, shuffle=True, pin_memory=True, num_workers=8, drop_last=True)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=1, drop_last=True)
        return dataloader

    def validation_step(self, batch, batch_idx):
        out_im = []
        gt_im = []

        with torch.no_grad():
            ground_truth = batch['img'][0].cuda()
            batch['coords'] = batch['coords'][0].cuda()
            batch['img'] = batch['img'][0].cuda()
            model_output = self.forward(batch)

            # Rearrange
            img_out = model_output['model_out']
            img_out = img_out.permute(0, 2, 1) # BS, C, PS
            img_out = img_out.view(ground_truth.shape)

        out_im = img_out
        gt_im = ground_truth

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

        out_im = torch.stack(out_channels)
        gt_im = torch.stack(gt_channels)
        psnr_ = psnr(out_im, gt_im)
        self.log('val_psnr', psnr_)

        if batch_idx < 4:
            if self.current_epoch < 2:
                self.logger.experiment.add_image(
                    f'gt/{batch_idx}', gt_im.clamp(0, 1), self.global_step)
            self.logger.experiment.add_image(
                            f'est/{batch_idx}', out_im.clamp(0, 1), self.global_step)


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
	monitor='val_psnr',
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
