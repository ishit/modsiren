"""Train an image model."""
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

import utils
import dataset
from hparams import hparams
import model

seed_everything(42)


class FunctionalImagingModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.train_dataset = dataset.celebAGeneralization(
            filenames=args.train_filenames, num_images=1e8)
        self.val_dataset = dataset.celebAGeneralization(
            filenames=args.val_filenames, num_images=1e8)

        print("Creating train codes")
        self.train_codes = torch.rand(
            len(self.train_dataset),
            hparams['cells'] ** 2, hparams['latent_dim']) * 0.001
        self.train_codes = torch.nn.Parameter(self.train_codes)

        if 'modsiren' in args.model:
            self.model = model.SimpleEncoderModSiren(
                in_features=2, out_features=hparams['channels'])
        else:
            self.model = \
                model.SimpleConvolutionalNeuralProcessImplicit2DHypernet(
                    in_features=2,
                    out_features=hparams['channels'])

    def forward(self, batch):
        batch['coords'] = batch['coords'].view(
            -1, self.train_dataset.cell_width ** 2, 2)
        idx = batch['idx']
        z_coords = self.train_codes[idx].view(-1, hparams['latent_dim'])
        batch['embedding'] = z_coords

        model_output = self.model(batch)

        return model_output

    def training_step(self, batch, batch_idx):
        ground_truth = batch['img']
        model_output = self.forward(batch)
        img_out = model_output['model_out'].view(
            hparams['batch_size'], -1, hparams['channels'],
            self.train_dataset.cell_width, self.train_dataset.cell_width)

        img_loss = torch.mean((img_out - ground_truth).pow(2))
        latent_loss = torch.mean(model_output['latent_vec'] ** 2)

        ground_truth = utils.unblockshaped_tensor(
            ground_truth, hparams['imsize'], hparams['imsize'])
        output = utils.unblockshaped_tensor(
            img_out, hparams['imsize'], hparams['imsize'])

        if not hparams['encoder']:
            loss = hparams['l2_weight'] * img_loss + \
                hparams['reg_weight'] * latent_loss
        else:
            loss = hparams['l2_weight'] * img_loss

        psnr_ = utils.psnr(output, ground_truth)

        logs = {'train/loss': loss,
                'train/l2': img_loss,
                'train/latent_loss': latent_loss,
                'train/psnr': psnr_}

        idx = batch['idx'].tolist()

        for i in range(4):
            if i in idx:
                j = idx.index(i)
                gt_show = ground_truth[j]
                out_show = output[j]

                self.logger.experiment.add_image(
                    f'train/gt/{i}',
                    torch.clamp(gt_show, 0, 1), self.global_step)
                self.logger.experiment.add_image(
                    f'train/est/{i}',
                    torch.clamp(out_show, 0, 1), self.global_step)

        for k, v in logs.items():
            self.log(k, v)

        return loss

    def validation_step(self, batch, batch_idx):

        if hparams['encoder']:
            ground_truth = batch['img']
            model_output = self.forward(batch)
            img_out = model_output['model_out'].view(
                hparams['batch_size'], -1, hparams['channels'],
                self.val_dataset.cell_width, self.val_dataset.cell_width)

            img_loss = torch.mean((img_out - ground_truth).pow(2))

            ground_truth = utils.unblockshaped_tensor(
                ground_truth, hparams['imsize'], hparams['imsize'])
            output = utils.unblockshaped_tensor(
                img_out, hparams['imsize'], hparams['imsize'])

            psnr_ = utils.psnr(output, ground_truth)

            logs = {'val_l2': img_loss,
                    'val_psnr': psnr_}

            idx = batch['idx'].tolist()

            for i in range(4):
                if i in idx:
                    j = idx.index(i)
                    gt_show = ground_truth[j]
                    out_show = output[j]
                    self.logger.experiment.add_image(
                        f'val/gt/{i}',
                        torch.clamp(gt_show, 0, 1), self.current_epoch)
                    self.logger.experiment.add_image(
                        f'val/est/{i}',
                        torch.clamp(out_show, 0, 1), self.current_epoch)

            for k, v in logs.items():
                self.log(k, v)

        else:
            torch.set_grad_enabled(True)
            ground_truth = batch['img']
            test_codes = torch.zeros(
                1, hparams['cells'] ** 2, hparams['latent_dim']) * 0.001
            test_codes = test_codes.clone().detach().requires_grad_(True)

            test_optimizer = torch.optim.Adam([test_codes], lr=1e-3)

            batch['img'] = batch['img'].view(
                -1, 1, self.train_dataset.cell_width,
                self.train_dataset.cell_width)
            batch['coords'] = batch['coords'].view(
                -1, self.train_dataset.cell_width ** 2, 2)

            for i in range(1):
                idx = batch['idx']
                z_coords = test_codes[idx].view(-1, hparams['latent_dim'])
                batch['embedding'] = z_coords

                test_optimizer.zero_grad()
                model_output = self.forward(batch, batch_idx)
                img_out = model_output['model_out'].view(
                    hparams['batch_size'], -1, 1,
                    self.train_dataset.cell_width,
                    self.train_dataset.cell_width)

                img_loss = torch.mean(
                    (img_out - ground_truth).pow(2))
                img_loss.backward()
                print('Test loss:', img_loss.item())
                test_optimizer.step()
                self.zero_grad()

            ground_truth = utils.unblockshaped_tensor(
                ground_truth, hparams['imsize'], hparams['imsize'])
            output = utils.unblockshaped_tensor(
                img_out, hparams['imsize'], hparams['imsize'])
            loss = img_loss
            psnr_ = utils.psnr(output, ground_truth)

            logs = {'test/loss': loss,
                    'test/l2': img_loss,
                    'test/psnr': psnr_}

            idx = batch['idx'].tolist()
            for i in range(1):
                if i in idx:
                    j = idx.index(i)
                    gt_show = ground_truth[j]
                    out_show = output[j]
                    plt.imsave(
                        f'gt{i}.png', gt_show[0].detach().cpu().numpy(),
                        cmap='gray')
                    plt.imsave(
                        f'test_{i}.png', out_show[0].detach().cpu().numpy(),
                        cmap='gray')

            return {"loss": loss, "log": logs}

    def configure_optimizers(self):
        train_optimizer = torch.optim.Adam(
            list(self.parameters()), lr=hparams["lr"])
        return train_optimizer

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset, batch_size=hparams['batch_size'],
            shuffle=hparams['shuffle'], pin_memory=True, num_workers=8)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset, batch_size=hparams['batch_size'], shuffle=False,
            pin_memory=True, num_workers=8)
        return dataloader


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', default='modsiren')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--train_filenames', type=str, required=True)
    parser.add_argument('--val_filenames', type=str, required=True)
    args = parser.parse_args()

    logger = TensorBoardLogger(
        save_dir="logs",
        name=args.model)

    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        save_top_k=2,
        monitor='val_psnr',
        mode='max',
        period=1,
    )

    module = FunctionalImagingModule()
    trainer = Trainer(max_epochs=hparams['epochs'],
                      logger=logger,
                      fast_dev_run=False,
                      gpus=[args.gpu],
                      # Val after every 10k images
                      val_check_interval=min(10000, len(
                          module.train_dataset)) // hparams['batch_size'],
                      profiler=True,
                      checkpoint_callback=checkpoint_callback)
    trainer.fit(module)
