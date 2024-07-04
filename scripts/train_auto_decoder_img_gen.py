from modsiren.datasets_bak import TexturesDataset
from pytorch_lightning.callbacks import Callback
from modsiren.models_bak import LocalMLP, GlobalMLP, NeuralProcessImplicit2DHypernet, AE
from modsiren.utils import psnr, hypo_weight_loss, latent_loss
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

class FunctionalImagingModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.train_dataset = TexturesDataset(
            root=config.root_dir, filenames=config.train_filenames, num_images=config.image.train_size, imsize=config.image.size, \
                    channels=config.image.channels, crop_size=config.image.crop_size, random_crop=False)

        ## Function Signature
        # def __init__(self, in_features, out_features, hidden_layers=4, hidden_features=256, latent_dim=None, 
                    # synthesis_activation=None, modulation_activation=None, embedding=None, N_freqs=None, 
                    # encoder=False, patch_res=patch_res):

        self.train_codes = torch.randn(config.training.batch_size, config.latent.dim) * config.latent.var_factor
        self.train_codes = torch.nn.Parameter(self.train_codes)

        if config.model.decoder:
            self.model = AE(out_features=config.image.channels, latent_dim=config.latent.dim, \
                                patch_res = (config.image.size, config.image.size))

        else:
            if not config.model.hyper:
                self.model = LocalMLP(in_features=2, out_features=config.image.channels, \
                                        hidden_layers=config.model.layers, hidden_features=config.model.width, \
                                        latent_dim = config.latent.dim, \
                                        synthesis_activation=config.model.synthesis_activation, \
                                        modulation_activation=config.model.modulation_activation, \
                                        embedding=config.model.embedding, \
                                        concat=config.model.concat, \
                                        N_freqs=config.model.N_freqs, \
                                        encoder = config.model.encoder, \
                                        encoder_type = config.model.encoder_type, \
                                        patch_res = (config.image.size, config.image.size)) 
            else:
                self.model = NeuralProcessImplicit2DHypernet(in_features=2, out_features=config.image.channels, \
                                        hidden_layers=config.model.layers, hidden_features=config.model.width, \
                                        latent_dim = config.latent.dim, \
                                        encoder = config.model.encoder, \
                                        encoder_type = config.model.encoder_type, \
                                        patch_res = (config.image.size, config.image.size))

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        ground_truth = batch['img']
        self.train_codes.data = batch['latent']
        batch['embedding'] = self.train_codes

        model_output = self.forward(batch)

        # Rearrange
        img_out = model_output['model_out']
        img_out = img_out.permute(0, 2, 1) # BS, C, PS
        img_out = img_out.view(
            config.training.batch_size, config.image.channels, config.image.size, config.image.size)

        img_loss = torch.mean((img_out - ground_truth).pow(2))
        loss = img_loss

        if config.model.hyper:
            loss = loss + config.loss.hypo_lam * hypo_weight_loss(model_output) 

        loss = loss + config.loss.reg_lam * latent_loss(model_output)
        psnr_ = psnr(img_out, ground_truth)

        logs = {'loss': loss,
                'l2': img_loss,
                'psnr': psnr_,
                'sample_idx': batch['idx']}


        if self.global_step % config.logging.scalars_step == 0:
            self.logger.experiment.add_scalar('psnr', psnr_, self.global_step)
            self.logger.experiment.add_scalar('img_loss', img_loss, self.global_step)

        if self.global_step % config.logging.image_step == 0:
            out_w_gt = torch.cat([img_out[:config.logging.image_vis_count], ground_truth[:config.logging.image_vis_count]], dim=0)
            img_grid = make_grid(out_w_gt, nrow=config.logging.image_vis_count)
            self.logger.experiment.add_image(
                            f'train_w_gt', img_grid.clamp(0, 1), self.global_step)

        return logs

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        self.train()

        val_codes = torch.randn(config.training.batch_size, config.latent.dim) * config.latent.var_factor
        val_codes = val_codes.cuda()
        val_codes.requires_grad = True

        val_optimizer = torch.optim.Adam([val_codes], lr=config.training.val_lr)
        ground_truth = batch['img'].float().cuda()

        for i in range(config.training.val_steps):
            val_optimizer.zero_grad()
            batch['embedding'] = val_codes

            model_output = self.forward(batch)

            # Rearrange
            img_out = model_output['model_out']
            img_out = img_out.permute(0, 2, 1) # BS, C, PS
            img_out = img_out.view(
                config.training.batch_size, config.image.channels, config.image.size, config.image.size)

            img_loss = torch.mean((img_out - ground_truth).pow(2))
            loss = img_loss

            if config.model.hyper:
                loss = loss + config.loss.hypo_lam * hypo_weight_loss(model_output) 

            loss = loss + config.loss.reg_lam * latent_loss(model_output)
            loss.backward()
            val_optimizer.step()

        torch.set_grad_enabled(False)
        self.eval()

        out_im = img_out
        gt_im = ground_truth

        self.zero_grad()
        psnr_ = psnr(img_out, ground_truth)
        self.log('val_psnr', psnr_)

        out_w_gt = torch.cat([out_im[:config.logging.image_vis_count],
                              gt_im[:config.logging.image_vis_count]], dim=0)
        img_grid = make_grid(out_w_gt, nrow=config.logging.image_vis_count)

        if batch_idx < 1:
            self.logger.experiment.add_image(
                f'est_w_gt', img_grid.clamp(0, 1), self.global_step)

        if not config.model.decoder:
            # 2x evaluation
            ground_truth_2x = batch['img_2x'].cuda()
            batch['coords'] = batch['coords_2x']
            batch['embedding'] = val_codes
            model_output_2x = self.forward(batch)

            img_out_2x = model_output_2x['model_out']
            img_out_2x = img_out_2x.permute(0, 2, 1)  # BS, C, PS
            img_out_2x = img_out_2x.view(ground_truth_2x.shape)

            out_im_2x = img_out_2x
            gt_im_2x = ground_truth_2x

            psnr_2x = psnr(out_im_2x, gt_im_2x)
            self.log('val_psnr_2x', psnr_2x)
            out_w_gt_2x = torch.cat(
                [out_im_2x[:config.logging.image_vis_count], gt_im_2x[:config.logging.image_vis_count]], dim=0)
            img_grid_2x = make_grid(
                out_w_gt_2x, nrow=config.logging.image_vis_count)
            if batch_idx < 1:
                self.logger.experiment.add_image(
                    f'est_w_gt_2x', img_grid_2x.clamp(0, 1), self.global_step)

    def configure_optimizers(self):
        train_optimizer = torch.optim.Adam(
            list(self.parameters()) + [self.train_codes], lr=config.training.lr)
        return train_optimizer

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset, batch_size=config.training.batch_size, shuffle=True, pin_memory=True, num_workers=8, drop_last=True)
        return dataloader

    def val_dataloader(self):
        val_dataset = TexturesDataset(
            root=config.root_dir, filenames=config.val_filenames, num_images=config.image.val_size, imsize=config.image.size, \
                    channels=config.image.channels, crop_size=config.image.crop_size, random_crop=False)
        dataloader = DataLoader(
            val_dataset, batch_size=config.training.batch_size, shuffle=False, pin_memory=True, num_workers=1, drop_last=True)
        return dataloader

class SaveFeatures(Callback):

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        pl_module.train_dataset.train_codes[batch['idx']] = pl_module.train_codes.cpu().data.clone() 

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        save_latent_path = os.path.join(config.log_dir, config.exp_name, f'version_{pl_module.logger.version}', 'latent.pt')
        torch.save(pl_module.train_dataset.train_codes, save_latent_path)

    def on_validation_epoch_start(self, trainer, pl_module):
        torch.set_grad_enabled(True)
        pl_module.train()

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
                      profiler=False,
                      checkpoint_callback=checkpoint_callback,
                      callbacks=[SaveFeatures()])
    trainer.fit(module)
