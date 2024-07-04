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

from modsiren.datasets_bak import Video, SingleVideoDataset
from modsiren.models_bak import LocalMLP, GlobalMLP
from modsiren.utils import psnr, video_unblockshaped_wc

import matplotlib.pyplot as plt

seed_everything(42)

class FunctionalVideoModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.vid_dataset = SingleVideoDataset(config.vid_path, config.video.channels, config.video.cell_size) 

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


        if not config.model.encoder:
            self.train_codes = torch.randn(
                    np.prod(self.vid_dataset.n_voxels), config.latent.dim) * config.latent.var_factor
            self.train_codes = torch.nn.Parameter(self.train_codes)

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

        if (self.global_step) % config.logging.image_step == 0:
            self.log_images()

        return loss

    def log_images(self, test=False):
        x_max, y_max, z_max = self.vid_dataset.n_voxels

        if not test: 
            time_idx = [0, 5, 10]
        else:
            time_idx = [i for i in range(50)]

        rows = np.arange(self.vid_dataset.n_voxels[1])
        cols = np.arange(self.vid_dataset.n_voxels[2])
        coords = np.transpose([np.tile(cols, len(rows)), np.repeat(rows, len(cols))])
        idx = coords[:, 1] * x_max + coords[:, 0] * x_max * y_max
        total_psnr = 0
        frame_cntr = 0

        with torch.no_grad():
            for j in time_idx:
                ret = []
                ret_gt = []
                for i in idx.tolist():
                    batch = self.vid_dataset[j + i]
                    latent_idx = batch['idx']
                    ground_truth = batch['img']
                    model_input = {}
                    coords = torch.FloatTensor(batch['coords']).unsqueeze(0).cuda()
                    model_input['coords'] = coords
                    global_coords = torch.FloatTensor(batch['global_coords']).unsqueeze(0).cuda()
                    model_input['global_coords'] = global_coords

                    if not config.model.encoder:
                        model_input['embedding'] = self.train_codes[latent_idx].cuda()
                    model_input['img'] = torch.FloatTensor(batch['img']).unsqueeze(0).cuda().permute(0, 4, 1, 2, 3)
                    model_output = self.forward(model_input)
                    output = model_output['model_out'].view(ground_truth.shape)
                    ret.append(output.detach().cpu().numpy())
                    ret_gt.append(ground_truth)

                ret = np.stack(ret)
                ret_gt = np.stack(ret_gt)
                vid = video_unblockshaped_wc(ret, self.vid_dataset.shape[1], self.vid_dataset.shape[2])
                vid_gt = video_unblockshaped_wc(ret_gt, self.vid_dataset.shape[1], self.vid_dataset.shape[2])

                if not test:
                    out_im = vid[0].transpose(2, 0, 1)
                    gt_im = vid_gt[0].transpose(2, 0, 1)
                    psnr_ = psnr(np.clip(out_im, 0, 1), gt_im)
                    total_psnr += psnr_
                    self.logger.experiment.add_image(
                                    f'est_{j}', np.clip(out_im, 0, 1), self.global_step)
                    self.logger.experiment.add_image(
                            f'gt_{j}', np.clip(gt_im, 0, 1), 0)
                else:
                    for t in range(vid.shape[0]):
                        out_im = vid[t]
                        gt_im = vid_gt[t]

                        # out_im = self.bilinear_blending(out_im)
                        # gt_im = self.bilinear_blending(gt_im)

                        psnr_ = psnr(np.clip(out_im, 0, 1), gt_im)
                        total_psnr += psnr_
                        frame_cntr += 1
                        print(frame_cntr)
                        plt.imsave(f'pe_test_vid/{frame_cntr:03d}.png', np.clip(out_im, 0, 1))
        
        if not test:
            self.log('val_psnr', total_psnr / len(time_idx), self.global_step)
            self.logger.experiment.add_scalar('val_psnr', total_psnr / len(time_idx), self.global_step)
        else:
            print(total_psnr / 250)

    def configure_optimizers(self):
        train_optimizer = torch.optim.Adam(
            list(self.parameters()), lr=config.training.lr)
        return train_optimizer

    def train_dataloader(self):
        dataloader = DataLoader(
            self.vid_dataset, shuffle=True, batch_size=self.hparams.batch_size, pin_memory=True, num_workers=8, drop_last=True)
        return dataloader

    def bilinear_blending(self, im):
        cell_width = self.vid_dataset.cell_size[1]

        def linear_blending(im):
            indcs_1 = np.arange(0, im.shape[1], 2 * cell_width)
            all_indcs_1 = []
            for i in range(cell_width):
                all_indcs_1.append(indcs_1 + i)
            all_indcs_1 = np.sort(np.concatenate(all_indcs_1))

            all_indcs_2 = all_indcs_1 + cell_width
            all_indcs_2 = all_indcs_2[:-cell_width]

            im_1 = im[:, all_indcs_1]
            im_2 = im[:, all_indcs_2]

            im_2 = np.append(im_2, im_1[:, -cell_width//2:], 1)
            im_2 = np.append(im_1[:, :cell_width//2], im_2, 1)

            xx, _ = np.meshgrid(np.arange(im_1.shape[1]), np.arange(im_1.shape[0]))
            xx = xx % cell_width
            xx = xx / (cell_width)
            xx = 1 - 2 * np.abs(xx - 0.5)
            xx = np.expand_dims(xx, -1)
            xx = np.repeat(xx, 3, -1)

            return im_1 * xx + im_2 * (1 - xx)

        im = linear_blending(im)
        im = linear_blending(im.transpose(1, 0, 2)).transpose(1, 0, 2)
        return im

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', default='config.yml')
    args = parser.parse_args()

    # Load config file
    with open(args.config, 'r') as f:
        config = AttrDict(yaml.safe_load(f))

    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        save_top_k=3,
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
                      profiler=False,
                      checkpoint_callback=checkpoint_callback,
                      auto_scale_batch_size='power')

    module = FunctionalVideoModule()
    # trainer.tune(module)
    trainer.fit(module)
