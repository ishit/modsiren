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
from modsiren.datasets_bak import VideoDataset, SingleVideoDataset
from modsiren.models_bak import (AE, GlobalMLP, LocalMLP,
                                 NeuralProcessImplicit2DHypernet)
from modsiren.utils import hypo_weight_loss, latent_loss, psnr, video_unblockshaped_wc, video_unblockshaped_tensor_wc
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

        ## Function Signature
        # def __init__(self, root, filenames, num_videos, cell_size, crop_size, random_crop=True):
        
        self.train_dataset = VideoDataset(
            root=config.root_dir, filenames=config.train_filenames, num_videos=config.video.train_size, cell_size=config.video.size,
            crop_size=config.video.crop_size, random_crop=True)

        # Function Signature
        # def __init__(self, in_features, out_features, hidden_layers=4, hidden_features=256, latent_dim=None,
        # synthesis_activation=None, modulation_activation=None, embedding=None, N_freqs=None,
        # encoder=False, patch_res=patch_res):

        if not config.model.hyper:
            self.model = LocalMLP(in_features=3, out_features=config.video.channels,
                                  hidden_layers=config.model.layers, hidden_features=config.model.width,
                                  latent_dim=config.latent.dim,
                                  synthesis_activation=config.model.synthesis_activation,
                                  modulation_activation=config.model.modulation_activation,
                                  concat=config.model.concat,
                                  embedding=config.model.embedding,
                                  freq_scale=config.model.freq_scale,
                                  N_freqs=config.model.N_freqs,
                                  encoder=config.model.encoder,
                                  encoder_type=config.model.encoder_type,
                                  patch_res=config.video.size)
        else:
            self.model = NeuralProcessImplicit2DHypernet(in_features=3, out_features=config.video.channels,
                                                         hidden_layers=config.model.layers, hidden_features=config.model.width,
                                                         latent_dim=config.latent.dim,
                                                         encoder=config.model.encoder,
                                                         encoder_type=config.model.encoder_type,
                                                          patch_res=config.video.size)

        self.hparams.batch_size = config.training.batch_size

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        ground_truth = batch['img']
        model_output = self.forward(batch)

        # Rearrange
        img_out = model_output['model_out']
        img_out = img_out.permute(0, 2, 1)  # BS, C, PS
        img_out = img_out.view(
            config.training.batch_size, config.video.channels, config.video.size[0], config.video.size[1], config.video.size[2]) 

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
            out_w_gt = torch.cat([img_out[:config.logging.image_vis_count, :, 3],
                ground_truth[:config.logging.image_vis_count, :, 3]], dim=0)
            # print(out_w_gt.shape)
            # exit()
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
            self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, pin_memory=True, num_workers=8, drop_last=True)
        return dataloader

    def val_dataloader(self):
        val_dataset = VideoDataset(
            root=config.root_dir, filenames=config.val_filenames, num_videos=config.video.train_size, cell_size=config.video.size,
            crop_size=config.video.crop_size, random_crop=False)

        dataloader = DataLoader(
            val_dataset, batch_size=config.training.batch_size, shuffle=False, pin_memory=True, num_workers=1, drop_last=True)
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

        out_w_gt = torch.cat([out_im[:config.logging.image_vis_count, :, 3],
            gt_im[:config.logging.image_vis_count, :, 3]], dim=0)
        img_grid = make_grid(out_w_gt, nrow=config.logging.image_vis_count)

        if batch_idx < 1:
            self.logger.experiment.add_image(
                f'est_w_gt', img_grid.clamp(0, 1), self.global_step)

    def log_images(self, test=False):
        vid_dataset = SingleVideoDataset('data/bikes.npy', config.video.channels, config.video.size) 
        x_max, y_max, z_max = vid_dataset.n_voxels

        if not test: 
            time_idx = [0, 5, 10]
        else:
            time_idx = [i for i in range(x_max)]

        rows = np.arange(vid_dataset.n_voxels[1])
        cols = np.arange(vid_dataset.n_voxels[2])
        coords = np.transpose([np.tile(cols, len(rows)), np.repeat(rows, len(cols))])
        idx = coords[:, 1] * x_max + coords[:, 0] * x_max * y_max
        total_psnr = 0
        frame_cntr = 0

        with torch.no_grad():
            for j in time_idx:
                ret = []
                ret_gt = []
                for i in idx.tolist():
                    batch = vid_dataset[j + i]
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
                vid = video_unblockshaped_wc(ret, vid_dataset.shape[1], vid_dataset.shape[2])
                vid_gt = video_unblockshaped_wc(ret_gt, vid_dataset.shape[1], vid_dataset.shape[2])

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
                        plt.imsave(f'test/{frame_cntr:03d}.png', np.clip(out_im, 0, 1))
        
        if not test:
            self.log('val_psnr', total_psnr / len(time_idx), self.global_step)
            self.logger.experiment.add_scalar('val_psnr', total_psnr / len(time_idx), self.global_step)
        else:
            print(total_psnr / 250)

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
                          check_val_every_n_epoch=1,
                          profiler=False,
                          checkpoint_callback=checkpoint_callback)
                          # callbacks=[SetupCallback()])

        module = FunctionalImagingModule()
        trainer.fit(module)
    # best_ckpt_dir = os.path.join(
        # config.log_dir, config.exp_name, f'version_{version}', 'checkpoints')

    # print(best_ckpt_dir)

    # trainer = Trainer(fast_dev_run=False,
                      # gpus=[config.gpu])

    # ckpts = sorted(glob.glob(best_ckpt_dir + '/*.ckpt'), key=os.path.getmtime)
    # if len(ckpts) == 0:
        # print("no checkpoints, aborting test")
    # else:
        # best_ckpt_path = ckpts[-1].strip()
        # print(best_ckpt_path)

        # module = FunctionalImagingModule.load_from_checkpoint(
            # checkpoint_path=best_ckpt_path).cuda()
        # module.log_images(test=True)
        # trainer.test(model=module, ckpt_path=best_ckpt_path)
