import sys
sys.path.append('./') 
sys.path.insert(0, './policy/Diffusion-Policy') 

import torch  
import os

import hydra
import omegaconf
from omegaconf import OmegaConf 
from pathlib import Path
from collections import deque, defaultdict
from tasks import *
import traceback

import yaml
from datetime import datetime
import dill
from policy_lightning.env_runner.dp_runner import DPRunner
from planner.motionplanner import PandaArmMotionPlanningSolver

import gymnasium as gym
import numpy as np
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from utils.wrappers.record import RecordEpisodeMA

import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = ""
    """The environment ID of the task you want to simulate"""

    config: str = "configs/table/strike_cube_hard.yaml"
    """Configuration to build scenes, assets and agents."""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "rgb"
    """Observation mode"""

    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = None
    """Robot UID(s) to use. Can be a comma separated list of UIDs or empty string to have no agents. If not given then defaults to the environments default robot"""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""

    reward_mode: Optional[str] = None
    """Reward mode"""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run."""

    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = "pd_joint_pos"
    """Control mode"""

    render_mode: str = "sensors"
    """Render mode"""

    shader: str = "default"
    """Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""

    record_dir: Optional[str] = './testvideo/{env_id}'
    """Directory to save recordings"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = False
    """Disable verbose output."""

    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = 10000
    """Seed(s) for random actions and simulator. Can be a single integer or a list of integers. Default is None (no seeds)"""

    data_num: int = 150
    """The number of episode data used for training the policy"""

    checkpoint_num: int = 50
    """The number of training epoch of the checkpoint"""

    record_dir: Optional[str] = './eval_video/{env_id}'
    """Directory to save recordings"""

    max_steps: int = 50
    """Maximum number of steps to run the simulation"""

    ckpt: str = 'policy_lightning/outputs/DP2/2025.06.27.01.04.44_2a_strike_cube_hard/checkpoints/last.ckpt'
    """Checkpoint Path"""

def get_policy(checkpoint, output_dir, device):
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cfg = OmegaConf.create(cfg)

    # configure model
    model: LightningModule = hydra.utils.instantiate(cfg.policy)
    model.load_state_dict(payload['state_dict'])

    device = torch.device(device)
    policy = model.to(device)
    policy.eval()

    return policy

class DP:
    def __init__(self, task_name, checkpoint_num: int, data_num: int, ckpt_path, id: int = 0):
        self.policy = get_policy(ckpt_path, None, 'cuda:0')
        self.runner = DPRunner(output_dir=None)

    def init_runner(self):
        self.runner = DPRunner(output_dir=None)

    def update_obs(self, observation):
        self.runner.update_obs(observation)
    
    def get_action(self, observation=None):
        action = self.runner.get_action(self.policy, observation)
        return action

    def get_last_obs(self):
        return self.runner.obs[-1]

def get_model_input(observation, agent_pos, agent_num):
    head_cam_dict = {}
    for agent_id in range(agent_num):
    # for agent_id in range(1):
        camera_name = 'head_camera' + '_agent' + str(agent_id)
        head_cam = np.moveaxis(observation['sensor_data'][camera_name]['rgb'].squeeze(0).cpu().numpy(), -1, 0) / 255   
        head_cam_dict.update({f'head_cam_{agent_id}': head_cam})
    head_cam_dict.update({f'agent_pos': agent_pos})
    return head_cam_dict


def main(args: Args):
    np.set_printoptions(suppress=True, precision=5)
    verbose = 0
    if isinstance(args.seed, int):
        args.seed = [args.seed]
    if args.seed is not None:
        np.random.seed(args.seed[0])
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False
    env_id = args.env_id
    if env_id == "":
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            env_id = config['task_name'] + '-rf'
            print(env_id)
    env_kwargs = dict(
        config=args.config,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        enable_shadow=True,
        parallel_in_single_scene=parallel_in_single_scene,
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))
    
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_file = f"logs/dp_global_{env_id}_{args.data_num}_{args.checkpoint_num}_{timestamp}.txt"

    # Load multi dp policy when load multi model
    # dp_models = []
    # for i in range(agent_num):
    #     dp_models.append(DP(env_id, args.checkpoint_num, args.data_num, id=i))
    # Load multi dp policy when load one model
    dp = DP(env_id, args.checkpoint_num, args.data_num, args.ckpt)
    total_success = 0
    total_num = 0
    now_success = 0
    for now_seed in range(args.seed[0], args.seed[0] + 100):
        dp.init_runner()
        env: BaseEnv = gym.make(env_id, **env_kwargs)
        print("Current eval seed: ", now_seed)
        total_num += 1
        now_success = 0
        np.random.seed(now_seed)
        record_dir = args.record_dir + '_dp_global_' + str(timestamp) + '/' + str(now_seed)
        if record_dir:
            record_dir = record_dir.format(env_id=env_id)
            env = RecordEpisodeMA(env, record_dir, info_on_video=False, save_trajectory=False, max_steps_per_video=30000000)
        raw_obs, _ = env.reset(seed=now_seed)
        planner = PandaArmMotionPlanningSolver(
            env,
            debug=False,
            vis=verbose,
            base_pose=[agent.robot.pose for agent in env.agent.agents],
            visualize_target_grasp_pose=verbose,
            print_env_info=False,
            is_multi_agent=True
        )
        agent_num = planner.agent_num
        if now_seed is not None and env.action_space is not None:
            env.action_space.seed(now_seed)
        if args.render_mode is not None:
            viewer = env.render()
            if isinstance(viewer, sapien.utils.Viewer):
                viewer.paused = args.pause
            env.render()
        initial_qpos_list = []
        for id in range(agent_num):
            initial_qpos = raw_obs['agent'][f'panda-{id}']['qpos'].squeeze(0)[:-2].cpu().numpy()
            initial_qpos = np.append(initial_qpos, planner.gripper_state[id]) 
            initial_qpos_list.append(initial_qpos)
        initial_qpos_all = np.concatenate(initial_qpos_list)        # shape: [n*8]
        obs = get_model_input(raw_obs, initial_qpos_all, agent_num)
        dp.update_obs(obs)
        cnt = 0
        while True:
            if verbose:
                print("Iteration: ", cnt)
            cnt = cnt + 1
            if cnt > args.max_steps:
                break
            if cnt % 15 == 0:
                print("iter: ", cnt)
            action = dp.get_action()
            action_dict = defaultdict(list)
            action_step_dict = defaultdict(list)
            for id in range(agent_num):
                # action_list = dp_models[id].get_action()  # for local policy
                action_list = []  # collect per-step actions for agent id
                for t in range(len(action)):  # iterate over time steps
                    full_action = action[t]   # shape: [n*8]
                    agent_action = full_action[id * 8 : (id + 1) * 8]  # slice for this agent
                    action_list.append(agent_action)
                for i in range(6):
                    now_action = action_list[i]
                    raw_obs = env.get_obs()
                    if i == 0:
                        current_qpos = raw_obs['agent'][f'panda-{id}']['qpos'].squeeze(0)[:-2].cpu().numpy()
                    else:
                        current_qpos = action_list[i - 1][:-1]
                    path = np.vstack((current_qpos, now_action[:-1]))
                    try:
                        # important for speed of eval
                        times, position, right_vel, acc, duration = planner.planner[id].TOPP(path, 0.05, verbose=True)
                    except Exception as e:
                        print(f"Error occurred: {e}")
                        action_now = np.hstack([current_qpos, now_action[-1]])
                        action_dict[f'panda-{id}'].append(action_now)
                        action_step_dict[f'panda-{id}'].append(1)
                        continue
                    n_step = position.shape[0]
                    action_step_dict[f'panda-{id}'].append(n_step)
                    gripper_state = now_action[-1]
                    if n_step == 0:
                        action_now = np.hstack([current_qpos, gripper_state])
                        action_dict[f'panda-{id}'].append(action_now)
                    for j in range(n_step):
                        true_action = np.hstack([position[j], gripper_state])
                        action_dict[f'panda-{id}'].append(true_action)
            
            start_idx = []
            for id in range(agent_num):
                start_idx.append(0)
            for i in range(6):
                max_step = 0
                for id in range(agent_num):
                    max_step = max(max_step, action_step_dict[f'panda-{id}'][i])
                for j in range(max_step):
                    true_action = dict()
                    for id in range(agent_num):
                        now_step = min(j, action_step_dict[f'panda-{id}'][i] - 1)
                        true_action[f'panda-{id}'] = action_dict[f'panda-{id}'][start_idx[id] + now_step]
                    # action execute
                    observation, reward, terminated, truncated, info = env.step(true_action)
                if max_step == 0:
                    continue
                action_concat = []
                skip = 0
                for id in range(agent_num):
                    start_idx[id] += action_step_dict[f'panda-{id}'][i]
                    action_concat.append(true_action[f'panda-{id}'])
                if skip:
                    continue
                if action_concat:
                    final_action = np.concatenate(action_concat)  # shape: [n*8]
                    obs = get_model_input(observation, final_action, agent_num)
                    dp.update_obs(obs)
            info = env.get_info()
            # print("info", info)
            if args.render_mode is not None:
                env.render()
            if info['success'] == True:
                total_success += 1
                now_success = 1
                env.close()
                if record_dir:
                    print(f"Saving video to {record_dir}")
                print("success, step=", cnt)
                break
        with open(log_file, "a") as f:
            f.write(f"\n[Summary] Success Rate: {total_success}% / {total_num}\n")
            f.write(f"Current Seeds: {now_seed}, success: {now_success}\n")
        if now_success == 0:
            print("failed")
            env.close() 
        if record_dir:
            print(f"Saving video to {record_dir}")

if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
