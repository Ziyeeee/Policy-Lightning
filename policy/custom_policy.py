from typing import Dict
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from policy.base_policy import BasePolicy


class CustomPolicy(BasePolicy):
    def __init__(self,
                 optimazer_cfg: DictConfig,
                 scheduler_cfg: DictConfig,
                 # your custom parameters
        ):

        super().__init__(optimazer_cfg, scheduler_cfg)

        # your custom initialization code
        # self.model = ...

        

    def predict_action(self, obs_dict):
        # your custom action prediction code

        return {
            'action': ...,          # predicted action for eval
            'action_pref': ...,     # predicted action for val
        }

    def compute_loss(self, batch, **kwargs):
        # your custom loss computation code
        
        loss = ...
        return loss