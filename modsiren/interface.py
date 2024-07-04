"""Model and data interfaces."""
import torch as th
from torch.utils.data import DataLoader
import numpy as np

import pytorch_lightning as pl

from . import config
from . import datasets
from . import metrics
from . import models


class BaseModule(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()

        # Make sure we at least have the default conf
        self.conf = config.merge(config.default_config(), conf)
        self.save_hyperparameters()


class SingleInstanceTrainingModule(BaseModule):
    def __init__(self, conf, num_cells):
        super().__init__(conf)

        if self.conf.task == "single_instance":
            if self.conf.domain == "image":
                num_in = 2
                num_out = 3
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

        model_cls = getattr(models, conf.model)
        self.model = model_cls(
            num_cells,
            input_features=num_in,
            output_features=num_out,
            **conf.model_params)

        if self.conf.training.loss == "l2":
            self.im_loss = th.nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss: {self.conf.training.pixel_loss}")

        self.psnr = metrics.PSNR()
        self.val_psnr = metrics.PSNR()

    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        opt = th.optim.Adam(
            self.model.parameters(),
            lr=self.conf.training.lr,
            betas=(0.9, 0.999),
            eps=1e-8)

        return opt

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        ref = batch['ref']

        loss = th.nn.functional.mse_loss(out, ref)

        self.log('loss', loss)
        self.log("psnr", self.psnr(out, ref), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        return out

    def test_step(self, batch, batch_idx):
        """Do nothing here, our test logic is in a callback."""

    def on_pretrain_routine_end(self):
        self.print(f"Training model \"{self.conf.name}\" "
                   f"on dataset: {self.conf.dataset}")


class SingleInstanceDataModule(pl.LightningDataModule):
    """Dataset interface for the single instance tasks.

    Args:
        conf(Omniconf): the global configuration.
    """

    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        if self.conf.task == "single_instance":
            if self.conf.domain == "image":
                self.trainset = datasets.SingleImagePatches(
                    self.conf.dataset, **self.conf.data)
                # No validation in the single instance case
                self.valset = self.trainset
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def num_cells(self):
        return len(self.trainset)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def _worker_init_fn(self, worker_id):
        np.random.seed(worker_id)
        th.manual_seed(worker_id)

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.conf.training.bs,
            num_workers=self.conf.training.workers,
            worker_init_fn=self._worker_init_fn,
            drop_last=False,
            shuffle=True,
            pin_memory=True)

    def val_dataloader(self):
        if self.valset is None:
            return None

        return DataLoader(
            self.valset,
            batch_size=self.conf.training.val_bs,
            num_workers=self.conf.training.workers,
            worker_init_fn=self._worker_init_fn,
            drop_last=False,
            pin_memory=True)
