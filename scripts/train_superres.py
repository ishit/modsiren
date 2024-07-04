from modsiren.datasets_bak import SuperRes
from modsiren.models_bak import LocalMLP, GlobalMLP, NeuralProcessImplicit2DHypernet, AE
from modsiren.utils import psnr
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, Resize
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
from lpips_pytorch import lpips

seed_everything(42)

class FunctionalImagingModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.train_dataset = SuperRes(
            root=config.root_dir, filenames=config.train_filenames, num_images=config.image.train_size, imsize=config.image.size, \
                    channels=config.image.channels, crop_size=config.image.crop_size, random_crop=False)
        self.val_dataset = SuperRes(
            root=config.root_dir, filenames=config.val_filenames, num_images=config.image.val_size, imsize=config.image.size, \
                    channels=config.image.channels, crop_size=config.image.crop_size, random_crop=False)

        ## Function Signature
        # def __init__(self, in_features, out_features, hidden_layers=4, hidden_features=256, latent_dim=None, 
                    # synthesis_activation=None, modulation_activation=None, embedding=None, N_freqs=None, 
                    # encoder=False, patch_res=patch_res):

        if config.model.decoder:
            self.model = AE(out_features=config.image.channels, latent_dim=config.latent.dim, \
                                patch_res = (config.image.size // 4, config.image.size // 4))

        else:
            if not config.model.hyper:
                self.model = LocalMLP(in_features=2, out_features=config.image.channels, \
                                        hidden_layers=config.model.layers, hidden_features=config.model.width, \
                                        latent_dim = config.latent.dim, \
                                        synthesis_activation=config.model.synthesis_activation, \
                                        modulation_activation=config.model.modulation_activation, \
                                        embedding=config.model.embedding, \
                                        N_freqs=config.model.N_freqs, \
                                        encoder = config.model.encoder, \
                                        encoder_type = config.model.encoder_type, \
                                        patch_res = (config.image.size // 4, config.image.size // 4)) 
            else:
                self.model = NeuralProcessImplicit2DHypernet(in_features=2, out_features=config.image.channels, \
                                        hidden_layers=config.model.layers, hidden_features=config.model.width, \
                                        latent_dim = config.latent.dim, \
                                        encoder = config.model.encoder, \
                                        encoder_type = config.model.encoder_type, \
                                        patch_res = (config.image.size // 4, config.image.size // 4))

    def forward(self, batch):
        resize_transform = Resize(config.image.size // 4)
        batch['img'] = resize_transform(batch['img'])
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        ground_truth = batch['img']

        down_transform = Resize(config.image.size // 4)
        up_transform = Resize(config.image.size)
        bicubic = up_transform(down_transform(ground_truth))

        model_output = self.forward(batch)

        # Rearrange
        img_out = model_output['model_out']
        img_out = img_out.permute(0, 2, 1) # BS, C, PS
        diff = img_out.view(
            config.training.batch_size, config.image.channels, config.image.size, config.image.size)
        img_out = diff + bicubic

        img_loss = torch.mean((img_out - ground_truth).pow(2))

        psnr_ = psnr(img_out, ground_truth)
        # lpips_loss = lpips(img_out.float(), ground_truth.float(), net_type='alex', version='0.1')
        # loss = img_loss + 1e-2 * lpips_loss
        loss = img_loss

        logs = {'loss': loss,
                'l2': img_loss,
                'psnr': psnr_}

        if self.global_step % config.logging.scalars_step == 0:
            self.logger.experiment.add_scalar('psnr', psnr_, self.global_step)
            self.logger.experiment.add_scalar('img_loss', img_loss, self.global_step)

        if self.global_step % config.logging.image_step == 0:
            out_w_gt = torch.cat([diff[:config.logging.image_vis_count] / 2 + 0.5, \
                    bicubic[:config.logging.image_vis_count], \
                    img_out[:config.logging.image_vis_count], \
                    ground_truth[:config.logging.image_vis_count]], dim=0)
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

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset, batch_size=config.training.batch_size, shuffle=False, pin_memory=True, num_workers=1, drop_last=True)
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset, batch_size=config.training.batch_size, shuffle=False, pin_memory=True, num_workers=1, drop_last=True)
        return dataloader

    def validation_step(self, batch, batch_idx):
        # 1x evaluation
        ground_truth = batch['img']
        down_transform = Resize(config.image.size // 4)
        up_transform = Resize(config.image.size)
        bicubic = up_transform(down_transform(ground_truth))

        model_output = self.forward(batch)

        img_out = model_output['model_out']
        img_out = img_out.permute(0, 2, 1) # BS, C, PS
        diff = img_out.view(ground_truth.shape)
        img_out = diff + bicubic

        out_im = img_out
        gt_im = ground_truth

        psnr_ = psnr(out_im, gt_im)
        self.log('val_psnr', psnr_)

        out_w_gt = torch.cat([diff[:config.logging.image_vis_count] / 2 + 0.5, \
                bicubic[:config.logging.image_vis_count], \
                out_im[:config.logging.image_vis_count], \
                gt_im[:config.logging.image_vis_count]], dim=0)
        img_grid = make_grid(out_w_gt, nrow=config.logging.image_vis_count)

        if batch_idx < 1:
            self.logger.experiment.add_image(
                            f'est_w_gt', img_grid.clamp(0, 1), self.global_step)

        # if not config.model.decoder:
            # # 2x evaluation
            # ground_truth_2x = batch['img_2x']
            # batch['coords'] = batch['coords_2x']
            # batch['img'] = batch['img']
            # model_output_2x = self.forward(batch)

            # img_out_2x = model_output_2x['model_out']
            # img_out_2x = img_out_2x.permute(0, 2, 1) # BS, C, PS
            # img_out_2x = img_out_2x.view(ground_truth_2x.shape)

            # out_im_2x = img_out_2x
            # gt_im_2x = ground_truth_2x

            # psnr_2x = psnr(out_im_2x, gt_im_2x)
            # self.log('val_psnr_2x', psnr_2x)
            # out_w_gt_2x = torch.cat([out_im_2x[:config.logging.image_vis_count], gt_im_2x[:config.logging.image_vis_count]], dim=0)
            # img_grid_2x = make_grid(out_w_gt_2x, nrow=config.logging.image_vis_count)
            # if batch_idx < 1:
                # self.logger.experiment.add_image(
                                # f'est_w_gt_2x', img_grid_2x.clamp(0, 1), self.global_step)


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
                      check_val_every_n_epoch=1,
                      profiler=True,
                      checkpoint_callback=checkpoint_callback)
    trainer.fit(module)
