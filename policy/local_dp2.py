from typing import Dict
from omegaconf import DictConfig
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from policy.base_policy import BasePolicy
from model.diffusion.conditional_unet1d import ConditionalUnet1D
from model.diffusion.mask_generator import LowdimMaskGenerator
from model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from common.pytorch_util import dict_apply


class DP2(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: MultiImageObsEncoder,
            optimazer_cfg: DictConfig,
            scheduler_cfg: DictConfig,
            agent_num: int,
            share_obs_encoder: bool,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            # parameters passed to step
            **kwargs):
        super().__init__(optimazer_cfg, scheduler_cfg)

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps
            
        # model = ConditionalUnet1D(
        #     input_dim=input_dim,
        #     local_cond_dim=None,
        #     global_cond_dim=global_cond_dim,
        #     diffusion_step_embed_dim=diffusion_step_embed_dim,
        #     down_dims=down_dims,
        #     kernel_size=kernel_size,
        #     n_groups=n_groups,
        #     cond_predict_scale=cond_predict_scale
        # )
        # self.model = model
        self.model_list = nn.ModuleList([
            ConditionalUnet1D(
                input_dim=action_dim,
                local_cond_dim=None,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=down_dims,
                kernel_size=kernel_size,
                n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            )
            for _ in range(agent_num)
        ])

        self.obs_encoders = nn.ModuleList([
            deepcopy(obs_encoder) for _ in range(agent_num)
        ]) if not share_obs_encoder else obs_encoder
        # self.obs_encoder = obs_encoder
        
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.agent_num = agent_num
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask, agent_id,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model_list[agent_id]
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory

    def get_local_agent_from_batch(self, batch, agent_id):
        return {
            k.rsplit('_', 1)[0]: v
            for k, v in batch.items()
            if k.endswith(f"_{agent_id}")
        }

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict
        result = {}
        nobs = self.normalizer(obs_dict)
        for agent_id in range(self.agent_num):
            # 1. get agent obs
            agent_nobs = self.get_local_agent_from_batch(nobs, agent_id)
            obs_encoder = self.obs_encoders[agent_id] if isinstance(self.obs_encoders, nn.ModuleList) else self.obs_encoders

            value = next(iter(agent_nobs.values()))
            B, To = value.shape[:2]
            T = self.horizon
            Da = self.action_dim
            Do = self.obs_feature_dim
            To = self.n_obs_steps

            device = value.device
            dtype = value.dtype

            local_cond = None
            global_cond = None

            if self.obs_as_global_cond:
                this_nobs = dict_apply(agent_nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
                nobs_features = obs_encoder(this_nobs)
                global_cond = nobs_features.reshape(B, -1)
                cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
                cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            else:
                this_nobs = dict_apply(agent_nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
                nobs_features = obs_encoder(this_nobs)
                nobs_features = nobs_features.reshape(B, To, -1)
                cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
                cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
                cond_data[:,:To,Da:] = nobs_features
                cond_mask[:,:To,Da:] = True

            # 2. Run sampling for this agent
            nsample = self.conditional_sample(
                cond_data, 
                cond_mask,
                agent_id,
                local_cond=local_cond,
                global_cond=global_cond,
                **self.kwargs
            )

            naction_pred = nsample[...,:Da]
            action_pred = self.normalizer[f'action_{agent_id}'].unnormalize(naction_pred)

            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

            result[f'action_{agent_id}'] = action
            result[f'action_pred_{agent_id}'] = action_pred  # optional

        return result

    def compute_loss(self, batch, **kwargs):
        assert 'valid_mask' not in batch
        total_loss = 0.0

        nobs = self.normalizer(batch['obs'])

        for agent_id in range(self.agent_num):
            model = self.model_list[agent_id]
            obs_encoder = self.obs_encoders[agent_id] if isinstance(self.obs_encoders, nn.ModuleList) else self.obs_encoders
            
            agent_nobs = self.get_local_agent_from_batch(nobs, agent_id)
            agent_action = batch[f'action_{agent_id}']
            nactions = self.normalizer[f'action_{agent_id}'].normalize(agent_action)
            batch_size = nactions.shape[0]
            horizon = nactions.shape[1]

            local_cond = None
            global_cond = None
            trajectory = nactions
            cond_data = trajectory

            if self.obs_as_global_cond:
                this_nobs = dict_apply(agent_nobs, lambda x: x[:, :self.n_obs_steps,...].reshape(-1, *x.shape[2:]))
                nobs_features = obs_encoder(this_nobs)
                global_cond = nobs_features.reshape(batch_size, -1)
            else:
                this_nobs = dict_apply(agent_nobs, lambda x: x.reshape(-1, *x.shape[2:]))
                nobs_features = obs_encoder(this_nobs)
                nobs_features = nobs_features.reshape(batch_size, horizon, -1)
                cond_data = torch.cat([nactions, nobs_features], dim=-1)
                trajectory = cond_data.detach()

            condition_mask = self.mask_generator(trajectory)
            noise = torch.randn(trajectory.shape, device=trajectory.device)
            bsz = trajectory.shape[0]
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (bsz,), device=trajectory.device
            ).long()
            noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
            loss_mask = ~condition_mask
            noisy_trajectory[condition_mask] = cond_data[condition_mask]

            pred = model(noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond)

            pred_type = self.noise_scheduler.config.prediction_type
            target = noise if pred_type == 'epsilon' else trajectory
            loss = F.mse_loss(pred, target, reduction='none')
            loss = loss * loss_mask.type(loss.dtype)
            loss = reduce(loss, 'b ... -> b (...)', 'mean').mean()

            total_loss += loss

        final_loss = total_loss / self.agent_num
        return final_loss   
    
    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('val/loss', loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
        obs_dict = batch['obs']
        
        total_mse = 0.0
        result = self.predict_action(obs_dict)
        for key, value in result.items():
            if key.startswith('action_pred'):
                pred_action = result[key]
                gt_action = batch[key.replace('_pred', '')]
                mse = F.mse_loss(pred_action, gt_action)
                total_mse += mse
        total_mse = total_mse / self.agent_num
        self.log('val/pred_action_mse', total_mse, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        return loss
