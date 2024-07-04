"""Evaluation metrics."""
import torch as th
import pytorch_lightning as pl


class PSNR(pl.metrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("psnr", default=th.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=th.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: th.Tensor, target: th.Tensor):
        assert preds.shape == target.shape

        preds = th.clamp(preds, 0, 1)

        count = th.tensor(preds.shape[0])
        mse = (preds-target).square().mean(-1).mean(-1).mean(-1)
        psnr = -10.0*th.log10(mse)
        # average per-image psnr
        self.count = self.count + count
        self.psnr = self.psnr + (psnr.sum(0) - count.float()*self.psnr) / \
            self.count.float()

    def compute(self):
        return self.psnr
