import glob
import time
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
from modsiren.datasets_bak import ShapeDataset
from modsiren.models_bak import (AE, GlobalMLP, LocalMLP,
                                 NeuralProcessImplicit2DHypernet, SineMLP)
from modsiren.utils import hypo_weight_loss, latent_loss, psnr, convert_sdf_samples_to_ply
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
        self.train_dataset = ShapeDataset(
            root=config.root_dir, filenames=config.train_filenames, num_shapes=config.shape.train_size,
            n_voxels=config.shape.n_voxels, sample_size=config.training.sample_size)

        # Function Signature
        # def __init__(self, in_features, out_features, hidden_layers=4, hidden_features=256, latent_dim=None,
        # synthesis_activation=None, modulation_activation=None, embedding=None, N_freqs=None,
        # encoder=False, patch_res=patch_res):

        if not config.model.hyper:
            self.model = LocalMLP(in_features=3, out_features=1,
                                  hidden_layers=config.model.layers, hidden_features=config.model.width,
                                  latent_dim=config.latent.dim,
                                  synthesis_activation=config.model.synthesis_activation,
                                  modulation_activation=config.model.modulation_activation,
                                  concat=config.model.concat,
                                  embedding=config.model.embedding,
                                  freq_scale=config.model.freq_scale,
                                  N_freqs=config.model.N_freqs,
                                  encoder=False,
                                  encoder_type=None,
                                  patch_res=())
        else:
            self.model = NeuralProcessImplicit2DHypernet(in_features=3, out_features=1,
                                                         hidden_layers=config.model.layers, hidden_features=config.model.width,
                                                         latent_dim=config.latent.dim,
                                                         encoder=False,
                                                         encoder_type=None,
                                                         patch_res=())

        self.train_codes = torch.randn(len(
            self.train_dataset), config.shape.n_voxels ** 3, config.latent.dim) * config.latent.var_factor
        self.train_codes = torch.nn.Parameter(self.train_codes)
        self.l1_loss = torch.nn.L1Loss(reduction="mean")

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        ground_truth = batch['sdf'].clamp(-config.shape.clamp, config.shape.clamp)

        batch['embedding'] = self.train_codes[batch['shape_idx']
                                              .view(-1).long(), batch['voxel_idx'].view(-1).long()]
        batch['embedding'] = batch['embedding'].view(
            config.training.batch_size, config.training.sample_size, -1)

        model_output = self.forward(batch)
        est_sdf = model_output['model_out'].clamp(-config.shape.clamp, config.shape.clamp)
        sdf_loss = self.l1_loss(est_sdf, ground_truth)

        loss = sdf_loss

        if config.model.hyper:
            loss = loss + config.loss.hypo_lam * hypo_weight_loss(model_output)

        latent_loss_ = latent_loss(
            model_output) * config.loss.reg_lam
        loss = loss + latent_loss_

        self.log('loss', loss)
        self.log('sdf_loss', sdf_loss)
        self.log('latent_loss', latent_loss_)

        return loss

    def configure_optimizers(self):
        train_optimizer = torch.optim.Adam(
            [
                {'params': self.model.parameters(), 'lr': config.training.model_lr},
                {'params': [self.train_codes], 'lr': config.training.latent_lr}
            ]
        )
        return train_optimizer

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset, batch_size=config.training.batch_size, shuffle=False, pin_memory=True, num_workers=1, drop_last=True)
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=1, drop_last=True)
        return dataloader

    def test_step(self, batch, batch_idx):
        start = time.time()
        file_id = self.file_ids[batch_idx].strip().split('/')[-1].split('.')[-2]

        ply_filename = f'{config.log_dir}/test/{config.exp_name}/{file_id}'
        os.makedirs(f'{config.log_dir}/test/', exist_ok=True)
        os.makedirs(f'{config.log_dir}/test/{config.exp_name}', exist_ok=True)

        N = 256
        max_batch = 64**3
        offset = None
        scale = None

        # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
        voxel_origin = [-1, -1, -1]
        voxel_size = 2.0 / (N - 1)

        overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
        samples = torch.zeros(N ** 3, 4)

        # transform first 3 columns
        # to be the x, y, z index
        samples[:, 2] = overall_index % N
        samples[:, 1] = (overall_index.long() // N) % N
        samples[:, 0] = ((overall_index.long() // N) // N) % N

        # transform first 3 columns
        # to be the x, y, z coordinate
        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
        samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
        samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
        num_samples = N ** 3

        samples.requires_grad = False
        head = 0

        print(f'Processing shape: {file_id}')
        while head < num_samples:
            print(f'{file_id}: {head}')
            sample_subset = samples[head: min(
                head + max_batch, num_samples), 0:3].cuda()

            model_input = {}
            model_input['coords'] = sample_subset
            shape_idx = torch.ones(sample_subset.shape[0]) * batch_idx 
            model_input['embedding'] = self.train_codes[shape_idx.view(-1).long(), 0, :]
            model_input['embedding'] = model_input['embedding'].view(shape_idx.shape[0], -1)
            model_output = self.forward(model_input)
            pred_sdf = model_output['model_out'].clamp(-config.shape.clamp, config.shape.clamp)
            pred_sdf[sample_subset.norm(dim=1) >= 1] = config.shape.clamp
            pred_sdf[torch.abs(sample_subset[:, 1]) >= 0.9] = config.shape.clamp
            pred_sdf[torch.abs(sample_subset[:, 0]) >= 0.9] = config.shape.clamp

            samples[head: min(head + max_batch, num_samples), 3] = (
                pred_sdf
                .squeeze()  # .squeeze(1)
                .detach()
                .cpu()
            )
            head += max_batch

        sdf_values = samples[:, 3]
        sdf_values = sdf_values.reshape(N, N, N)

        end = time.time()
        print("sampling takes: %f" % (end - start))

        convert_sdf_samples_to_ply(
            sdf_values.data.cpu(),
            voxel_origin,
            voxel_size,
            ply_filename + ".ply",
            offset,
            scale,
        )


class SetupCallback(Callback):
    def on_test_epoch_start(self, trainer, pl_module):
        with open(config.train_filenames, 'r') as f:
            lines = f.readlines()
            pl_module.file_ids = lines

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
        monitor='loss',
        mode='min',
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
                          # check_val_every_n_epoch=5,
                          profiler=False,
                          checkpoint_callback=checkpoint_callback)

        module = FunctionalImagingModule()
        trainer.fit(module)

    best_ckpt_dir = os.path.join(
        # config.log_dir, config.exp_name, 'version_0', 'checkpoints')
        config.log_dir, config.exp_name, 'version_1', 'checkpoints')

    trainer = Trainer(fast_dev_run=False,
                      gpus=[config.gpu],
                      callbacks=[SetupCallback()])

    ckpts = sorted(glob.glob(best_ckpt_dir + '/*.ckpt'), key=os.path.getmtime)
    best_ckpt_path = ckpts[-1].strip()
    print(best_ckpt_path)

    module = FunctionalImagingModule.load_from_checkpoint(
        checkpoint_path=best_ckpt_path)
    trainer.test(model=module, ckpt_path=best_ckpt_path)
