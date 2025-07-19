from lightning import LightningModule
import numpy as np
from omegaconf import OmegaConf
import torch
import hydra
import dill
from collections import deque
from typing import Any, Dict
from common.pytorch_util import dict_apply

class DeployPolicy:
    """
    DeployPolicy provides a minimal interface for loading and running a trained policy with history support.
    Usage:
        policy = DeployPolicy(ckpt_path)
        policy.update_obs(obs)
        action = policy.get_action()
        policy.reset()
    """
    def __init__(self, ckpt_path: str):
        """
        Initialize and load the policy from checkpoint. Automatically detects history length.
        Args:
            ckpt_path (str): Path to the checkpoint file.
        """
        payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        cfg = payload['cfg']

        # Deploy policy
        cfg = OmegaConf.create(cfg)
        model: LightningModule = hydra.utils.instantiate(cfg.policy)
        model.load_state_dict(payload['state_dict'])
        self.policy = model
        self.policy.to(self.device)
        self.policy.eval()
        
        # Read history length from config (n_obs_steps or horizon)
        self.n_obs_steps = getattr(cfg, 'n_obs_steps', None)
        
        self.obs = deque(maxlen=self.n_obs_steps+1)
        self.action = deque(maxlen=8)

    def get_model_input(self, observation, agent_pos, agent_num):
        head_cam_dict = {}
        agent_pos_list = []
        for agent_id in range(agent_num):
            camera_name = 'head_camera' + '_agent' + str(agent_id)
            head_cam = np.moveaxis(observation['sensor_data'][camera_name]['rgb'].squeeze(0).cpu().numpy(), -1, 0) / 255   
            head_cam_dict.update({f'head_cam_{agent_id}': head_cam})
            agent_pos_i = agent_pos[agent_id * 8 : (agent_id + 1) * 8]
            agent_pos_list.append(agent_pos_i)
        head_cam_dict.update({f'agent_pos': np.concatenate(agent_pos_list, axis=-1)})
        return head_cam_dict

    def update_obs(self, obs: Dict[str, Any]):
        qpos_list = []
        agent_num = len(obs['agent'])
        # In panda arm, we use past action's gripper state as the current state
        # so we need to record the action history
        for id in range(agent_num):
            current_qpos = obs['agent'][f'panda-{id}']['qpos'].squeeze(0)[:-2].cpu().numpy()
            if len(self.action) == 0:
                # Initialize with open if no action history
                current_qpos = np.append(current_qpos, 1)
            else:
                current_action = self.action[0]         # get the first action in the history as robot's current state
                current_qpos = np.append(current_qpos, current_action[(id + 1) * 8 - 1])  # last action is gripper action
            qpos_list.append(current_qpos)
        qpos_all = np.concatenate(qpos_list)  # shape: [n*8]
        if len(self.action) != 0:
            self.action.popleft()  # pop the first action to keep the length of action history consistent
        obs = self.get_model_input(obs, qpos_all, agent_num)
        self.obs.append(obs)

    def get_action(self) -> Any:
        device, dtype = self.policy.device, self.policy.dtype
        obs = self.get_n_steps_obs() #

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
            action_dict = self.policy.predict_action(obs_dict_input)

        # device_transfer
        np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
        actions = np_action_dict['action'].squeeze(0)

        # record action as robot's state
        for action in actions:
            self.action.append(action)
        return actions

    def reset(self):
        """
        Reset the policy and history at the beginning of each episode.
        """
        self.obs = deque(maxlen=self.n_obs_steps+1)
    
    def get_n_steps_obs(self):
        assert(len(self.obs) > 0), 'no observation is recorded, please update obs first'

        result = dict()
        for key in self.obs[0].keys():
            result[key] = self.stack_last_n_obs(
                [obs[key] for obs in self.obs],
                self.n_obs_steps
            )

        return result

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