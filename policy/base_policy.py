from typing import Dict
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from lightning.pytorch import LightningModule

from model.common.normalizer import LinearNormalizer
from model.common.lr_scheduler import get_scheduler


class BasePolicy(LightningModule):
    def __init__(self,
            optimazer_cfg: DictConfig,
            scheduler_cfg: DictConfig,
        ):
        super().__init__()
        self.normalizer = LinearNormalizer()
        self.optimizer_cfg = optimazer_cfg
        self.scheduler_cfg = scheduler_cfg

    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Predict action for eval
        obs_dict: 
            Dict[str, torch.Tensor]
            Observations used for prediction.
        Returns:
            Dict[str, torch.Tensor]
            Must contain 'action' key with the predicted action.
            'action_pred' must have the same shape as ground truth action.
        """
        return {
            'action': torch.tensor([0], device=self.device, dtype=self.dtype),
            'action_pred': torch.tensor([0], device=self.device, dtype=self.dtype)  # [Optional] Used for val
        }

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute loss for training
        batch: 
            Dict[str, torch.Tensor]
            Batch of data containing observations and ground truth actions.
        Returns:
            torch.Tensor
            Loss value computed from the batch.
        """
        return torch.tensor(-1.0, device=self.device, dtype=self.dtype)
    
    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('val/loss', loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
        obs_dict = batch['obs']
        gt_action = batch['action']
        
        result = self.predict_action(obs_dict)
        pred_action = result['action_pred'] if 'action_pred' in result else result['action']
        mse = F.mse_loss(pred_action, gt_action)
        self.log('val/action_mse', mse, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            **self.optimizer_cfg, 
            params=self.parameters()
        )
        lr_scheduler = get_scheduler(
            self.scheduler_cfg.scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.scheduler_cfg.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }