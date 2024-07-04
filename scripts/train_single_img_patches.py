import glob
from modsiren.utils import *
from modsiren.datasets_bak import ExtendedSingleImagePatches, SingleImagePatches 
from modsiren.models_bak import LocalMLP, GlobalMLP, NeuralProcessImplicit2DHypernet
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
        patch_res = config.image.tile_size

        if config.model.local:
            self.train_dataset = ExtendedSingleImagePatches(config.im_path, imsize=config.image.size, cells=config.image.cells, \
                    channels=config.image.channels)
        else:
            self.train_dataset = SingleImagePatches(config.im_path, imsize=config.image.size, cells=config.image.cells, \
                    channels=config.image.channels)
        
        self.train_codes = torch.randn((config.image.cells * 2) ** 2, config.latent.dim) * config.latent.var_factor
        self.train_codes = torch.nn.Parameter(self.train_codes)

        # Function Signature
        # def __init__(self, in_features, out_features, hidden_layers=4, hidden_features=256, latent_dim=None, 
                    # synthesis_activation=None, modulation_activation=None, embedding=None, N_freqs=None, 
                    # encoder=False, patch_res=patch_res):
        if config.model.local:
            self.model = LocalMLP(in_features=2, out_features=config.image.channels, \
                                    hidden_layers=config.model.layers, hidden_features=config.model.width, \
                                    latent_dim = config.latent.dim, \
                                    synthesis_activation=config.model.synthesis_activation, \
                                    modulation_activation=config.model.modulation_activation, \
                                    concat=config.model.concat, \
                                    embedding=config.model.embedding, \
                                    freq_scale=config.model.freq_scale,\
                                    N_freqs=config.model.N_freqs, \
                                    encoder = config.model.encoder, \
                                    encoder_type=config.model.encoder_type,\
                                    patch_res = (patch_res, patch_res))
        else:
            self.model = GlobalMLP(in_features=2, out_features=config.image.channels, \
                                    hidden_layers=config.model.layers, hidden_features=config.model.width, \
                                    synthesis_activation=config.model.synthesis_activation, \
                                    embedding=config.model.embedding, \
                                    N_freqs=config.model.N_freqs, \
                                    freq_scale=config.model.freq_scale)

    def forward(self, batch):
        BS, PS, D = batch['coords'].shape
        ground_truth = batch['img']
        idx = batch['idx']
        
        if config.model.local:
            z_coords = self.train_codes[idx].view(-1, config.latent.dim) # BS, ZD
            batch['embedding'] = z_coords

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

    def log_images(self, mode='train'):
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
        psnr_ = psnr(out_im, gt_im)

        if mode == 'train':
            self.log('val_psnr', psnr_)
            self.logger.experiment.add_image(
                            'est', out_im.clamp(0, 1), self.global_step)
            if self.global_step < 2:
                self.logger.experiment.add_image(
                    'gt', gt_im.clamp(0, 1), self.global_step)

        else:
            with open(best_ckpt_dir + '/test_results.txt', 'w') as f:
                print_str = f'val_psnr: {psnr_}\n'
                print(print_str)
                f.write(print_str)
                im_id = config.im_path.strip().split('/')[-1].split('.')[-2]
                plt.imsave(f'{best_ckpt_dir}/test_{im_id}.png',
                           out_im.permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy())




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


    version = 0
    if config.training.mode == 'train':
        logger = TensorBoardLogger(
            save_dir=config.log_dir,
            name=config.exp_name)

        version = logger.version

        trainer = Trainer(max_epochs=config.training.epochs,
                          logger=logger,
                          fast_dev_run=False,
                          gpus=[config.gpu],
                          profiler=False,
                          checkpoint_callback=checkpoint_callback)

        module = FunctionalImagingModule()
        trainer.fit(module)

    best_ckpt_dir = os.path.join(
        config.log_dir, config.exp_name, f'version_{version}', 'checkpoints')

    print(best_ckpt_dir)

    ckpts = sorted(glob.glob(best_ckpt_dir + '/*.ckpt'), key=os.path.getmtime)
    if len(ckpts) == 0:
        print("no checkpoints, aborting test")
    else:
        best_ckpt_path = ckpts[-1].strip()
        print(best_ckpt_path)

        module = FunctionalImagingModule.load_from_checkpoint(
            checkpoint_path=best_ckpt_path).cuda()
        module.log_images(mode='test')
