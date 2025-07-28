import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.helpers import update_class_from_dict

from collections import OrderedDict
import json
import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    with open(os.path.join("legged_gym/logs", train_cfg.runner.experiment_name, args.load_run, "config.json"), "r") as f:
        d = json.load(f, object_pairs_hook= OrderedDict)
        update_class_from_dict(env_cfg, d, strict= True)
        update_class_from_dict(train_cfg, d, strict= True)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    env.max_episode_length = 750
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    print("stop_rew_log",stop_rew_log)
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, log_dir = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        save_cfg= False,
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach(),i, stop_rew_log)

        # if  0 < i < stop_rew_log:
        #     if infos["episode"]:
        #         num_episodes = torch.sum(env.reset_buf).item()
        #         if num_episodes>0:
        #             logger.log_rewards(infos["episode"], num_episodes)
        # elif i==stop_rew_log:
        #     logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
