"""Train a functional model to overfit to a single image."""
from pytorch_lightning.loggers import TensorBoardLogger, CometLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset
import torch
import os
import skvideo.datasets
import skvideo.io
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
from utils import *
from model import *
from dataset import SingleImage
from argparse import ArgumentParser

seed_everything(42)


class FunctionalImagingModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.channels = 1
        self.imsize = 512
        self.dataset = SingleImage(
            self.imsize, batch_size=256 * 256, channels=self.channels)

        if args.model == 'siren':
            self.img_model = SirenImg(
                in_features=2, out_features=self.channels)
        elif args.model == 'ffn':
            self.img_model = FFN(in_features=2, out_features=self.channels)
        elif args.model == 'pemlp':
            self.img_model = PEMLP(in_features=2, out_features=self.channels)
        elif args.model == 'reluMLP':
            self.img_model = ReLUMLP(in_features=2, out_features=self.channels)
        else:
            print('Wrong model type!')
            exit()

    def forward(self, batch):
        return self.img_model(batch)

    def training_step(self, batch, batch_idx):
        coords = batch['coords']
        gt = batch['gt']

        model_input = {}
        model_input['coords'] = coords.float()
        model_output = self.forward(model_input)

        pred = model_output['model_out']
        loss = ((gt - pred)**2).mean()
        psnr_ = psnr(pred, gt)

        self.log('l2', loss)
        self.log('psnr', psnr_)

        if self.global_step % 200 == 0:
            self.log_images()

        return loss

    def log_images(self):
        out_im = []
        gt_im = []
        dataloader = DataLoader(
            self.dataset, shuffle=False, batch_size=1)

        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                coords = batch['coords']
                gt = batch['gt']

                model_input = {}
                model_input['coords'] = coords.float().cuda()
                model_output = self.forward(model_input)
                output = model_output['model_out']

                pred = model_output['model_out']
                out_im.append(pred)
                gt_im.append(gt)

                if len(out_im) >= self.dataset.len:
                    break

        out_im = torch.cat(out_im, dim=0).reshape(
            self.imsize, self.imsize, self.channels).permute(2, 0, 1)
        gt_im = torch.cat(gt_im, dim=0).reshape(
            self.imsize, self.imsize, self.channels).permute(2, 0, 1)
        self.logger.experiment.add_image(
            'est', ((out_im + 1) / 2).clamp(0, 1), self.global_step)
        self.logger.experiment.add_image(
            'gt', ((gt_im + 1) / 2).clamp(0, 1), self.global_step)

        return

    def configure_optimizers(self):
        train_optimizer = torch.optim.Adam(
            list(self.parameters()), lr=1e-4)
        return train_optimizer

    def train_dataloader(self):
        dataloader = DataLoader(
            self.dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=32)
        return dataloader


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', default='reluMLP')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--latent_dim', type=int, default=1024)
    args = parser.parse_args()

    # checkpoint_callback = ModelCheckpoint(
    #     verbose=True,
    #     save_top_k=2,
    #     monitor='psnr',
    #     mode='max',
    #     period=1,
    # )

    # logger = TensorBoardLogger(
    #     save_dir="logs",
    #     name=args.model)

    trainer = Trainer(max_epochs=100,
                      logger=logger,
                      fast_dev_run=False,
                      gpus=[args.gpu],
                      profiler=True,
                      checkpoint_callback=checkpoint_callback)

    module = FunctionalImagingModule()
    trainer.fit(module)
