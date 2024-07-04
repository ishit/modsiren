import glob
import os
from argparse import ArgumentParser

import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import yaml
from attrdict import AttrDict
from modsiren.datasets_bak import TexturesDataset
from modsiren.models_bak import (AE, GlobalMLP, LocalMLP,
                                 NeuralProcessImplicit2DHypernet, SineMLP)
from modsiren.utils import hypo_weight_loss, latent_loss, psnr
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from skimage import filters
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from torchvision.utils import make_grid

seed_everything(42)


class FunctionalImagingModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.train_dataset = TexturesDataset(
            root=config.root_dir, filenames=config.train_filenames, num_images=config.image.train_size, imsize=config.image.size,
            channels=config.image.channels, crop_size=config.image.crop_size, random_crop=False)

        # Function Signature
        # def __init__(self, in_features, out_features, hidden_layers=4, hidden_features=256, latent_dim=None,
        # synthesis_activation=None, modulation_activation=None, embedding=None, N_freqs=None,
        # encoder=False, patch_res=patch_res):

        if config.model.decoder:
            self.model = AE(out_features=config.image.channels, latent_dim=config.latent.dim,
                            patch_res=(config.image.size, config.image.size))

        else:
            if not config.model.hyper:
                self.model = LocalMLP(in_features=2, out_features=config.image.channels,
                                      hidden_layers=config.model.layers, hidden_features=config.model.width,
                                      latent_dim=config.latent.dim,
                                      synthesis_activation=config.model.synthesis_activation,
                                      modulation_activation=config.model.modulation_activation,
                                      concat=config.model.concat, \
                                      embedding=config.model.embedding,
                                      freq_scale=config.model.freq_scale,
                                      N_freqs=config.model.N_freqs,
                                      encoder=config.model.encoder,
                                      encoder_type=config.model.encoder_type,
                                      patch_res=(config.image.size, config.image.size))
            else:
                self.model = NeuralProcessImplicit2DHypernet(in_features=2, out_features=config.image.channels,
                                                             hidden_layers=config.model.layers, hidden_features=config.model.width,
                                                             latent_dim=config.latent.dim,
                                                             encoder=config.model.encoder,
                                                             encoder_type=config.model.encoder_type,
                                                             patch_res=(config.image.size, config.image.size))

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        ground_truth = batch['img']
        model_output = self.forward(batch)

        # Rearrange
        img_out = model_output['model_out']
        img_out = img_out.permute(0, 2, 1)  # BS, C, PS
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
                'psnr': psnr_}

        if self.global_step % config.logging.scalars_step == 0:
            self.logger.experiment.add_scalar('psnr', psnr_, self.global_step)
            self.logger.experiment.add_scalar(
                'img_loss', img_loss, self.global_step)

        if self.global_step % config.logging.image_step == 0:
            out_w_gt = torch.cat([img_out[:config.logging.image_vis_count],
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
        val_dataset = TexturesDataset(
            root=config.root_dir, filenames=config.val_filenames, num_images=config.image.val_size, imsize=config.image.size,
            channels=config.image.channels, crop_size=config.image.crop_size, random_crop=False)
        dataloader = DataLoader(
            val_dataset, batch_size=config.training.batch_size, shuffle=False, pin_memory=False, num_workers=1, drop_last=True)
        return dataloader

    def test_dataloader(self):
        test_dataset = TexturesDataset(
            root=config.root_dir, filenames=config.test_filenames, num_images=config.image.test_size, imsize=config.image.size,
            channels=config.image.channels, crop_size=config.image.crop_size, random_crop=False)
        dataloader = DataLoader(
            test_dataset, batch_size=config.training.batch_size, shuffle=False, pin_memory=True, num_workers=1, drop_last=True)
        return dataloader

    def validation_step(self, batch, batch_idx):
        # 1x evaluation
        ground_truth = batch['img'].cuda()
        model_output = self.forward(batch)

        img_out = model_output['model_out']
        img_out = img_out.permute(0, 2, 1)  # BS, C, PS
        img_out = img_out.view(ground_truth.shape)

        out_im = img_out
        gt_im = ground_truth

        psnr_ = psnr(out_im, gt_im)
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

    def test_step(self, batch, batch_idx):
        ground_truth = batch['img'].cuda().float()
        model_output = self.forward(batch)
        img_out = model_output['model_out']
        img_out = img_out.permute(0, 2, 1)  # BS, C, PS
        img_out = img_out.view(ground_truth.shape)

        out_im = img_out
        gt_im = ground_truth

        psnr_ = psnr(out_im, gt_im)
        lpips_ = self.lpips_loss(out_im, gt_im)

        out_w_gt = torch.cat([out_im[:config.logging.image_vis_count],
                              gt_im[:config.logging.image_vis_count]], dim=0)
        img_grid = make_grid(out_w_gt, nrow=config.logging.image_vis_count)
        plt.imsave(f'{best_ckpt_dir}/test_{batch_idx}.png',
                   img_grid.permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy())

        self.log('test_psnr', psnr_)
        self.log('test_lpips', lpips_)

        if not config.model.decoder:
            ground_truth_2x = batch['img_2x'].cuda().float()
            batch['coords'] = batch['coords_2x']
            model_output_2x = self.forward(batch)

            img_out_2x = model_output_2x['model_out']
            img_out_2x = img_out_2x.permute(0, 2, 1)  # BS, C, PS
            img_out_2x = img_out_2x.view(ground_truth_2x.shape)

            out_im_2x = img_out_2x
            gt_im_2x = ground_truth_2x

            psnr_2x = psnr(out_im_2x, gt_im_2x)
            lpips_2x = self.lpips_loss(out_im_2x, gt_im_2x)
            self.log('test_psnr_2x', psnr_2x)
            self.log('test_lpips_2x', lpips_2x)
            out_w_gt_2x = torch.cat(
                [out_im_2x[:config.logging.image_vis_count], gt_im_2x[:config.logging.image_vis_count]], dim=0)
            img_grid_2x = make_grid(
                out_w_gt_2x, nrow=config.logging.image_vis_count)
            plt.imsave(f'{best_ckpt_dir}/test_2x_{batch_idx}.png',
                       img_grid_2x.permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy())

    def optim_latent(self, batch, batch_idx):
        ground_truth_2x = batch['img_2x'].cuda().float()
        # ground_truth_2x = batch['img_2x'][0].cuda().float()
        # batch['coords'] = batch['coords_2x']
        model_output_2x = self.forward(batch)

        # img_out_2x = model_output_2x['model_out']
        # img_out_2x = img_out_2x.permute(0, 2, 1)  # BS, C, PS
        # img_out_2x = img_out_2x.view(ground_truth_2x.shape)

        # out_im_2x = img_out_2x
        # gt_im_2x = ground_truth_2x

        # psnr_2x = psnr(out_im_2x, gt_im_2x)
        # self.log('test_prev_psnr_2x', psnr_2x)

        latent_var = model_output_2x['latent_vec'][:, 0, :]

        torch.set_grad_enabled(True)
        self.train()

        # self.model.load_state_dict(torch.load('/tmp/weights.pt'))

        val_codes = latent_var.clone()
        val_codes = val_codes.cuda()
        # val_codes.requires_grad = True

        val_optimizer = torch.optim.Adam([val_codes] + list(self.model.parameters()), lr=config.training.val_lr)
        # val_optimizer = torch.optim.Adam(list(self.model.parameters()), lr=config.training.val_lr)
        # val_optimizer = torch.optim.Adam(list(self.model.parameters()), lr=1e-4)
        # val_optimizer = torch.optim.Adam(list(self.model.synthesis_nw.parameters()), lr=1e-4)
        # val_optimizer = torch.optim.Adam(self.model.synthesis_nw.parameters(), lr=1e-4)
        # val_optimizer = torch.optim.Adam([val_codes], lr=config.training.val_lr)

        encoder = self.model.encoder
        self.model.encoder = None

        batch['embedding'] = val_codes
        # batch['coords'] = batch['coords'][0]
        batch['coords'] = batch['coords_2x']
        for i in range(config.training.val_steps):
            model_output_2x = self.forward(batch)

            # Rearrange
            img_out_2x = model_output_2x['model_out']
            # img_out_2x = img_out_2x.permute(0, 2, 1) # BS, C, PS
            # img_out_2x = img_out_2x.permute(1, 0) # BS, C, PS
            # img_out_2x = img_out_2x.view(ground_truth_2x.shape)
            # img_out_2x = self.model.synthesis_nw(batch['coords_2x'])
            img_out_2x = img_out_2x.permute(0, 2, 1) # BS, C, PS
            img_out_2x = img_out_2x.view(ground_truth_2x.shape)


            # img_loss = torch.mean((img_out_2x - ground_truth_2x).pow(2))
            img_loss = ((img_out_2x - ground_truth_2x)**2).mean()

            out_im_2x = img_out_2x
            gt_im_2x = ground_truth_2x

            psnr_2x = psnr(out_im_2x, gt_im_2x)
            # self.log(f'test_{i:03d}_psnr_2x', psnr_2x)

            loss = img_loss
            print(loss.item())
            val_optimizer.zero_grad()
            loss.backward()
            val_optimizer.step()

            # out_w_gt_2x = torch.cat(
                # [out_im_2x[:config.logging.image_vis_count], gt_im_2x[:config.logging.image_vis_count]], dim=0)
            # img_grid_2x = make_grid(
                # out_w_gt_2x, nrow=config.logging.image_vis_count)
            # plt.imsave(f'{best_ckpt_dir}/test_2x_{batch_idx}_{i:03d}.png',
                       # img_grid_2x.permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy())

        torch.set_grad_enabled(False)
        self.eval()

        # img_out_2x = model_output_2x['model_out']
        # img_out_2x = img_out_2x.permute(0, 2, 1)  # BS, C, PS
        # img_out_2x = img_out_2x.view(ground_truth_2x.shape)

        # out_im_2x = img_out_2x
        # gt_im_2x = ground_truth_2x

        # psnr_2x = psnr(out_im_2x, gt_im_2x)
        # # print(psnr_2x)

        # lpips_2x = self.lpips_loss(out_im_2x, gt_im_2x)
        # self.log('test_psnr_2x', psnr_2x)
        # self.log('test_lpips_2x', lpips_2x)
        # out_w_gt_2x = torch.cat(
            # [out_im_2x[:config.logging.image_vis_count], gt_im_2x[:config.logging.image_vis_count]], dim=0)
        # img_grid_2x = make_grid(
            # out_w_gt_2x, nrow=config.logging.image_vis_count)
        # plt.imsave(f'{best_ckpt_dir}/test_2x_{batch_idx}.png',
                   # img_grid_2x.permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy())

        self.model.encoder = encoder

class SetupCallback(Callback):

    def on_test_epoch_start(self, trainer, pl_module):
        # torch.save(pl_module.model.state_dict(), '/tmp/weights.pt')
        # print(pl_module.model.embed.B)
        pl_module.lpips_loss = lpips.LPIPS().cuda()

    def on_test_epoch_end(self, trainer, pl_module):
        logs = trainer.logged_metrics
        # with open(best_ckpt_dir + '/test_results.txt', 'w') as f:
            # for key, val in logs.items():
                # if not 'test' in key:
                    # continue
                # print_str = f'{key}: {val.item()}\n'
                # print(print_str)
                # f.write(print_str)


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

    if config.training.mode == 'train':
        logger = TensorBoardLogger(
            save_dir=config.log_dir,
            name=config.exp_name)

        trainer = Trainer(max_epochs=config.training.epochs,
                          logger=logger,
                          fast_dev_run=False,
                          gpus=[config.gpu],
                          check_val_every_n_epoch=5,
                          profiler=True,
                          checkpoint_callback=checkpoint_callback,
                          callbacks=[SetupCallback()])

        module = FunctionalImagingModule()
        trainer.fit(module)

    best_ckpt_dir = os.path.join(
        config.log_dir, config.exp_name, 'version_0', 'checkpoints')

    trainer = Trainer(fast_dev_run=False,
                      gpus=[config.gpu],
                      callbacks=[SetupCallback()])

    ckpts = sorted(glob.glob(best_ckpt_dir + '/*.ckpt'), key=os.path.getmtime)
    if len(ckpts) == 0:
        print("no checkpoints, aborting test")
    else:
        best_ckpt_path = ckpts[-1].strip()
        print(best_ckpt_path)

        module = FunctionalImagingModule.load_from_checkpoint(
            checkpoint_path=best_ckpt_path)
        trainer.test(model=module, ckpt_path=best_ckpt_path)
