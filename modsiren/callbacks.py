"""Callbacks."""
import torch as th
import pytorch_lightning as pl


class SingleImageVisualizationCallback(pl.Callback):
    """Logs images to Tensorboard.

    Args:
        logger(pl.TensorboardLogger): the logger to write images to.
        frequency(int): the number of epochs between updates
    """

    def __init__(self, logger, frequency=10):
        super().__init__()
        self.logger = logger
        self.tiles = []
        self.ref_tiles = []
        self.grid_dims = []
        self.frequency = frequency

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                batch_idx, dataloader_idx):
        """Display training image every epoch."""
        self.tiles.append(outputs)
        self.ref_tiles.append(batch["ref"])
        self.grid_dims = batch["grid_dims"][0]

    def on_validation_epoch_start(self, trainer, pl_module):
        self.tiles = []
        self.ref_tiles = []
        self.grid_dims = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.current_epoch % self.frequency != 0:
            return

        tiles = th.cat(self.tiles, 0)
        ref_tiles = th.cat(self.ref_tiles, 0)
        tsz = tiles.shape[-1]
        h = self.grid_dims[0]
        w = self.grid_dims[1]

        # Not a full validation pass
        if tiles.shape[0] != h*w:
            return

        tiles = tiles.view(h, w, 3,  tsz, tsz).permute(2, 0, 3, 1, 4)
        tiles = tiles.reshape(3, h*tsz, w*tsz)
        tiles = th.clamp(tiles, 0, 1)

        ref_tiles = ref_tiles.view(h, w, 3,  tsz, tsz).permute(2, 0, 3, 1, 4)
        ref_tiles = ref_tiles.reshape(3, h*tsz, w*tsz)
        ref_tiles = th.clamp(ref_tiles, 0, 1)

        self.logger.experiment.add_image(
            "output", tiles, global_step=pl_module.current_epoch)
        self.logger.experiment.add_image(
            "ref", ref_tiles, global_step=pl_module.current_epoch)
