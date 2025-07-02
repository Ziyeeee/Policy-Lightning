import torch
import os
import numpy as np
import hydra
from pathlib import Path
from collections import deque

import yaml
from datetime import datetime
import importlib
import dill
from argparse import ArgumentParser
from ..common.pytorch_util import dict_apply


class DPRunner:
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=300,
                 n_obs_steps=3,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 tqdm_interval_sec=5.0,
                 task_name=None,
    ):
        self.task_name = task_name
        self.eval_episodes = eval_episodes
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.obs = deque(maxlen=n_obs_steps+1)
        self.env = None

    def stack_last_n_obs(self, all_obs, n_steps):
        assert(len(all_obs) > 0)
        all_obs = list(all_obs)
        if isinstance(all_obs[0], np.ndarray):
            result = np.zeros((n_steps,) + all_obs[-1].shape, 
                dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = np.array(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        elif isinstance(all_obs[0], torch.Tensor):
            result = torch.zeros((n_steps,) + all_obs[-1].shape, 
                dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = torch.stack(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        else:
            raise RuntimeError(f'Unsupported obs type {type(all_obs[0])}')
        return result
    
    def reset_obs(self):
        self.obs.clear()

    def update_obs(self, current_obs):
        self.obs.append(current_obs)

    def get_n_steps_obs(self):
        assert(len(self.obs) > 0), 'no observation is recorded, please update obs first'

        result = dict()
        for key in self.obs[0].keys():
            result[key] = self.stack_last_n_obs(
                [obs[key] for obs in self.obs],
                self.n_obs_steps
            )

        return result

    def get_action(self, policy, observaton=None):
        device, dtype = policy.device, policy.dtype
        if observaton is not None:
            self.obs.append(observaton) # update
        obs = self.get_n_steps_obs() #
        # import pdb; pdb.set_trace()

        # create obs dict
        np_obs_dict = dict(obs)
        # device transfer
        obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
        # run policy
        with torch.no_grad():
            obs_dict_input = {}  # flush unused keys
            for key in obs_dict.keys():
                if key.startswith('head_cam') or key.startswith('agent_pos'):          
                    obs_dict_input[key] = obs_dict[key].unsqueeze(0)
            # import pdb; pdb.set_trace()
            action_dict = policy.predict_action(obs_dict_input)

        # device_transfer
        np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
        action_pred_list = []
        for key in np_action_dict.keys():
            if key.startswith('action_pred'):
                action_pred_list.append(np_action_dict[key])
        if action_pred_list:
            merged_action_pred = np.concatenate(action_pred_list, axis=-1)  # 或 axis=1, 视你的数据shape而定
            action = merged_action_pred.squeeze(0)
        else:
            action = np_action_dict['action'].squeeze(0)
        return action