import glob
import time
import os
from argparse import ArgumentParser

import trimesh
import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import yaml
from attrdict import AttrDict
from modsiren.datasets_bak import SingleShapeDataset
from modsiren.models_bak import (AE, GlobalMLP, LocalMLP,
                                 NeuralProcessImplicit2DHypernet, SineMLP)
from modsiren.utils import hypo_weight_loss, latent_loss, psnr, convert_sdf_samples_to_ply, compute_trimesh_chamfer
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
        if config.model.local:
            mode = 'modsiren'
        else:
            mode = 'siren'

        self.train_dataset = SingleShapeDataset(n_voxels=config.shape.n_voxels,\
                points_path=config.points_path,\
                sdf_path=config.sdf_path,\
                sample_size=config.training.sample_size,\
                n_points=config.training.total_points, \
                mode=mode)

        # Function Signature
        # def __init__(self, in_features, out_features, hidden_layers=4, hidden_features=256, latent_dim=None,
        # synthesis_activation=None, modulation_activation=None, embedding=None, N_freqs=None,
        # encoder=False, patch_res=patch_res):

        if config.model.local:
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
            self.model = GlobalMLP(in_features=3, out_features=1,
                                  hidden_layers=config.model.layers, hidden_features=config.model.width,
                                  synthesis_activation=config.model.synthesis_activation,
                                  embedding=config.model.embedding,
                                  N_freqs=config.model.N_freqs,
                                  freq_scale=config.model.freq_scale)


        self.train_codes = torch.randn(config.shape.n_voxels ** 3, config.latent.dim) * config.latent.var_factor
        self.train_codes = torch.nn.Parameter(self.train_codes)
        self.l1_loss = torch.nn.L1Loss(reduction="mean")

        self.gt_mesh = trimesh.load_mesh(config.mesh_path)
        # self.l1_loss = torch.nn.MSELoss(reduction="mean")

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        ground_truth = batch['sdf'].clamp(-config.shape.clamp, config.shape.clamp)

        # batch['embedding'] = self.train_codes[batch['voxel_idx'].view(-1).long(), :].unsqueeze(0)
        batch['embedding'] = self.train_codes[batch['voxel_idx'].squeeze(-1).long(), :]
        model_output = self.forward(batch)

        est_sdf = model_output['model_out'].view(ground_truth.shape).clamp(-config.shape.clamp, config.shape.clamp)
        sdf_loss = self.l1_loss(est_sdf, ground_truth)
        loss = sdf_loss

        if config.model.local:
            latent_loss_ = latent_loss(
                model_output) * config.loss.reg_lam
            loss = loss + latent_loss_
            self.log('latent_loss', latent_loss_)

        self.log('loss', loss)
        self.log('sdf_loss', sdf_loss)

        if self.global_step % config.logging.shape_log == 0 and self.global_step > 1:
            cd = self.render()
            self.logger.experiment.add_scalar('cd1', cd, self.global_step)
            self.log('cd', cd)

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
            self.train_dataset, batch_size=config.training.batch_size, shuffle=True, pin_memory=True, num_workers=0, drop_last=True)
        return dataloader

    def localize_points(self, coords, n_voxels):
        self.bins = torch.linspace(-1, 1, n_voxels + 1).cuda()
        bin_idx = torch.searchsorted(self.bins, coords) - 1
        bin_idx = bin_idx.clamp(0, config.shape.n_voxels - 1)
        latent_idx = bin_idx[..., :1] * config.shape.n_voxels * config.shape.n_voxels +\
            bin_idx[..., 1:2] * config.shape.n_voxels + bin_idx[..., 2:]
        T_coords = coords - self.bins[bin_idx]
        sidelen = self.bins[1] - self.bins[0]
        T_coords = T_coords / sidelen
        T_coords = T_coords - 0.5
        T_coords = 2 * T_coords
        return T_coords, latent_idx

    def render(self):
        start = time.time()
        ply_filename = os.path.join(config.log_dir, config.exp_name, f'version_{self.logger.version}', f'dragon_{config.shape.n_voxels}_{self.global_step}.ply')

        N = 256
        max_batch = 32**3
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

        while head < num_samples:
            sample_subset = samples[head: min(
                head + max_batch, num_samples), 0:3].cuda()

            model_input = {}
            T_coords, latent_idx = self.localize_points(sample_subset, config.shape.n_voxels)
            model_input['coords'] = T_coords.float()
            model_input['global_coords'] = sample_subset.float()
            model_input['embedding'] = self.train_codes[latent_idx[:, 0]]
            model_output = self.forward(model_input)

            pred_sdf = model_output['model_out']

            if config.model.local:
                mask = np.isin(latent_idx[:, 0].detach().cpu(
                ).numpy(), self.train_dataset.non_empty_voxel_idx)
                mask = np.logical_not(mask)
                pred_sdf[mask] = config.shape.clamp

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

        convert_sdf_samples_to_ply(
            sdf_values.data.cpu(),
            voxel_origin,
            voxel_size,
            ply_filename,
            offset,
            scale,
        )

        est_mesh = trimesh.load_mesh(ply_filename)
        cd = compute_trimesh_chamfer(self.gt_mesh, est_mesh, config.eval_points)

        return cd

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
                          checkpoint_callback=checkpoint_callback,
                          distributed_backend='dp')

        module = FunctionalImagingModule()
        trainer.fit(module)

    # best_ckpt_dir = os.path.join(
        # config.log_dir, config.exp_name, 'version_0', 'checkpoints')

    # trainer = Trainer(fast_dev_run=False,
                      # gpus=[config.gpu],
                      # callbacks=[SetupCallback()])

    # ckpts = sorted(glob.glob(best_ckpt_dir + '/*.ckpt'), key=os.path.getmtime)
    # best_ckpt_path = ckpts[-1].strip()
    # print(best_ckpt_path)

    # module = FunctionalImagingModule.load_from_checkpoint(
        # checkpoint_path=best_ckpt_path)
    # trainer.test(model=module, ckpt_path=best_ckpt_path)


