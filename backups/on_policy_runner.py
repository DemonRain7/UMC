# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
import rsl_rl.algorithms as algorithms
import rsl_rl.modules as modules
from modules.actor_critic import get_activation
from rsl_rl.env import VecEnv
from modules.transformers import BodyActor,Transformer
from modules.transformer_a1_rld import Env_Factor_Encoder,Adaptation_Encoder

def load_model(model, path):
    # 加载保存的模型状态
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {path}")
    return model


class OnPolicyRunner:
    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        actor_critic = modules.build_actor_critic(
            self.env,
            self.cfg["policy_class_name"],
            self.policy_cfg,
        ).to(self.device)

        alg_class = getattr(algorithms, self.cfg["algorithm_class_name"]) # PPO
        self.alg: algorithms.PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.log_interval = self.cfg.get("log_interval", 1)

        # Adption_Module Training Phase
        # print(self.policy_cfg)
        if self.policy_cfg['actor_type'] == "transformer_rma_phase2":
            self.adaptation_encoder = Adaptation_Encoder(input_dim=48, output_dim=120, hidden_dim=64, tsteps=self.num_steps_per_env).to(self.device)
            self.adaptation_optimizer = optim.Adam(self.adaptation_encoder.parameters(), lr=5e-4)

            env_factor = Env_Factor_Encoder(input_dim=187+17, output_dim=120) # Dim for A1-Walk
            env_factor_path = './legged_gym/logs/field_a1/new-abl-rld-phase1-2/env_factor_encoder_2500.pt'  # 模型保存路径
            self.env_factor = load_model(env_factor, env_factor_path).to(self.device)
            self.env_factor.eval()
            
            actor_net = Transformer(self.policy_cfg['embedding_dim'], dim_feedforward=self.policy_cfg['dim_feedforward'], nhead=self.policy_cfg['nheads'], nlayers=self.policy_cfg['nlayers'], mask_position= self.policy_cfg['mask_pos'],numbodies = self.policy_cfg['nbodies'])
            base_policy_actor = BodyActor(self.policy_cfg['env_name'], actor_net, embedding_dim=self.policy_cfg['embedding_dim'], action_dim=12, stack_time=False, global_input=self.policy_cfg['actor_type']=='mlp', mu_activation=get_activation(self.policy_cfg['mu_activation']), device=device,nbodies = self.policy_cfg['nbodies'])
            actor_path = './legged_gym/logs/field_a1/new-abl-rld-phase1-2/base_actor_2500.pt'  # 模型保存路径
            self.base_policy_actor = load_model(base_policy_actor, actor_path).to(self.device)
            self.base_policy_actor.eval()


        # Curr_50_rollback
        self.curr_50_rollback = None

        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rframebuffer = deque(maxlen=2000)
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = self.current_learning_iteration + num_learning_iterations
        tmp_iter = 0


        while self.current_learning_iteration < tot_iter:
            # print("self.current_learning_iteration", self.current_learning_iteration)
            # print("tmp_iter:", tmp_iter)
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    obs, critic_obs, rewards, dones, infos = self.rollout_step(obs, critic_obs, tmp_iter)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rframebuffer.extend(rewards[dones < 1].cpu().numpy().tolist())
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            losses, stats = self.alg.update(self.current_learning_iteration)
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None and self.current_learning_iteration % self.log_interval == 0:
                self.log(locals())
            if self.current_learning_iteration % self.save_interval == 0 and self.current_learning_iteration > start_iter:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
                if self.policy_cfg['actor_type'] == "transformer_rma_phase2":
                    if self.alg.actor_critic.actor.env_factor_encoder is not None and self.alg.actor_critic.actor.actor is not None:
                        self.save_env_factor_encoder(os.path.join(self.log_dir, 'env_factor_encoder_{}.pt'.format(self.current_learning_iteration)))
                        self.save_base_actor(os.path.join(self.log_dir, 'base_actor_{}.pt'.format(self.current_learning_iteration)))
            ep_infos.clear()
            self.current_learning_iteration = self.current_learning_iteration + 1
            tmp_iter = tmp_iter+1
        
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
        if self.policy_cfg['actor_type'] == "transformer_rma_phase2":
            if self.alg.actor_critic.actor.env_factor_encoder is not None and self.alg.actor_critic.actor.actor is not None:
                self.save_env_factor_encoder(os.path.join(self.log_dir, 'env_factor_encoder_{}.pt'.format(self.current_learning_iteration)))
                self.save_base_actor(os.path.join(self.log_dir, 'base_actor_{}.pt'.format(self.current_learning_iteration)))

    def learn_tfql(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rframebuffer = deque(maxlen=2000)
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = self.current_learning_iteration + num_learning_iterations
        tmp_iter = 0


        while self.current_learning_iteration < tot_iter:
            # print("self.current_learning_iteration", self.current_learning_iteration)
            # print("tmp_iter:", tmp_iter)
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    obs, critic_obs, rewards, dones, infos = self.rollout_step_tfql(obs, critic_obs, tmp_iter)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rframebuffer.extend(rewards[dones < 1].cpu().numpy().tolist())
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            losses, stats = self.alg.update_tfql(self.current_learning_iteration)
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None and self.current_learning_iteration % self.log_interval == 0:
                self.log(locals())
            if self.current_learning_iteration % self.save_interval == 0 and self.current_learning_iteration > start_iter:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
            ep_infos.clear()
            self.current_learning_iteration = self.current_learning_iteration + 1
            tmp_iter = tmp_iter+1
        
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def learn_adaption_encoder(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rframebuffer = deque(maxlen=2000)
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = self.current_learning_iteration + num_learning_iterations
        tmp_iter = 0

        # Intialize Zt
        zt_hat = torch.zeros(self.env.num_envs, 120, device=self.device)  # 初始的 zt_hat 可以是零向量，或者是根据需要进行初始化

        while self.current_learning_iteration < tot_iter:
            # print("self.current_learning_iteration", self.current_learning_iteration)
            # print("tmp_iter:", tmp_iter)
            start = time.time()

            # Record State-Action Pairs
            state_action_pairs = []
            
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):

                    obs, critic_obs, rewards, dones, infos = self.rollout_adaptation_module_step(obs, critic_obs, tmp_iter, zt_hat)
                    obs = obs.to(self.device) 
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rframebuffer.extend(rewards[dones < 1].cpu().numpy().tolist())
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                    # Append the SA pair
                    state_action_pairs.append((obs[:,:48]))

                # Get Env_Factor Zt
                zt = self.env_factor(obs[:,48:(48+187+17)])

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop

            zt_for_loss = zt.clone().detach().requires_grad_(True)  # 克隆并启用梯度

            # Pass state_action_pairs through Adaptation Encoder 
            state_action_pairs = torch.stack(state_action_pairs)  # Shape: [tsteps, batch_size, 48]
            # print("SA SHAPE:", state_action_pairs.shape)
            z_hat = self.adaptation_encoder(state_action_pairs)

            # Calculate loss (e.g., Mean Squared Error loss)
            loss = F.mse_loss(z_hat, zt_for_loss)

            # Print the loss value
            print(f"Current Adaptation Module loss: {loss.item()}")

            # Backpropagate and optimize
            self.adaptation_optimizer.zero_grad()
            loss.backward()
            self.adaptation_optimizer.step()

            zt_hat = z_hat.detach() 

            # Log results if necessary
            if self.writer is not None:
                self.writer.add_scalar('Adaption_Module_TrainingLoss', loss.item(), self.current_learning_iteration)


            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None and self.current_learning_iteration % self.log_interval == 0:
                self.log_adapt(locals())
            if self.current_learning_iteration % self.save_interval == 0 and self.current_learning_iteration > start_iter:
                if self.adaptation_encoder is not None:
                    self.save_adapt_module_encoder(os.path.join(self.log_dir, 'adapt_module_encoder_{}.pt'.format(self.current_learning_iteration)))
            ep_infos.clear()
            self.current_learning_iteration = self.current_learning_iteration + 1
            # print("Current Iteration: ",self.current_learning_iteration)
            tmp_iter = tmp_iter+1
        
        if self.adaptation_encoder is not None:
            self.save_adapt_module_encoder(os.path.join(self.log_dir, 'adapt_module_encoder_{}.pt'.format(self.current_learning_iteration)))



    def rollout_step(self, obs, critic_obs, tmp_curr_iter):
        actions = self.alg.act(obs, critic_obs)
        obs, privileged_obs, rewards, dones, infos = self.env.step(actions, tmp_curr_iter)
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
        self.alg.process_env_step(rewards, dones, infos)
        return obs, critic_obs, rewards, dones, infos

    def rollout_step_tfql(self, obs, critic_obs, tmp_curr_iter):
        self.curr_50_rollback = self.alg.storage.past_50_obs[-1]
        # print("self.curr_50_rollback:", self.curr_50_rollback.shape)
        self.curr_50_rollback = self.curr_50_rollback.to(self.device)
        self.alg.storage.past_50_obs = self.alg.storage.past_50_obs.to(self.device)
        actions = self.alg.act_tfql(obs, critic_obs, self.curr_50_rollback)
        obs, privileged_obs, rewards, dones, infos = self.env.step(actions, tmp_curr_iter)

        # update storage
        # 从0-49，0为最贴近的step
        # 1. 获取当前观测 obs_prop，形状是 [num_envs, 48]
        # print(self.alg.storage.past_50_obs.shape, self.alg.storage.past_50_obs[0:2])

        obs_prop = obs[:, :48]  

        # 2. 丢弃 self.alg.storage.past_50_obs 的最后一个 step（形状 [1, 50, num_envs, 48]），右移其他的部分
        self.alg.storage.past_50_obs[1:] = self.alg.storage.past_50_obs[:-1].clone()

        # 3. 更新第 0 个 step
        # 先移除第 0 个 step 里最后一个 obs
        tmp_49_obs = self.alg.storage.past_50_obs[0][:-1]  # shape: [49, num_envs, 48]

        # 4. 在第 0 个 step 的最前面插入 obs_prop
        self.alg.storage.past_50_obs[0] = torch.cat([obs_prop.unsqueeze(0), tmp_49_obs], dim=0)  # shape: [50, num_envs, 48]

        # print(self.alg.storage.past_50_obs.shape, self.alg.storage.past_50_obs[0:2])

        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
        self.alg.process_env_step(rewards, dones, infos)
        return obs, critic_obs, rewards, dones, infos


    def rollout_adaptation_module_step(self, obs, critic_obs, tmp_curr_iter, latent_vector):
        # Get Actor Actions
        with torch.no_grad():  # 不需要计算梯度
            actions = self.base_policy_actor(obs, latent_vector = latent_vector)
        obs, privileged_obs, rewards, dones, infos = self.env.step(actions, tmp_curr_iter)
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
        return obs, critic_obs, rewards, dones, infos

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, self.current_learning_iteration)
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.action_std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        for k, v in locs["losses"].items():
            self.writer.add_scalar("Loss/" + k, v.item(), self.current_learning_iteration)
        for k, v in locs["stats"].items():
            self.writer.add_scalar("Train/" + k, v.item(), self.current_learning_iteration)
        
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, self.current_learning_iteration)
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), self.current_learning_iteration)
        self.writer.add_scalar('Perf/total_fps', fps, self.current_learning_iteration)
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], self.current_learning_iteration)
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], self.current_learning_iteration)
        self.writer.add_scalar('Perf/gpu_allocated', torch.cuda.memory_allocated(self.device) / 1024 ** 3, self.current_learning_iteration)
        self.writer.add_scalar('Perf/gpu_occupied', torch.cuda.mem_get_info(self.device)[1] / 1024 ** 3, self.current_learning_iteration)
        self.writer.add_scalar('Train/mean_reward_each_timestep', statistics.mean(locs['rframebuffer']), self.current_learning_iteration)
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), self.current_learning_iteration)
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), self.current_learning_iteration)
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {self.current_learning_iteration}/{locs['tot_iter']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs["losses"]['value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs["losses"]['surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
                        )
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs["losses"]['value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs["losses"]['surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
                        )

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (self.current_learning_iteration + 1 - locs["start_iter"]) * (
                               locs['tot_iter'] - self.current_learning_iteration):.1f}s\n""")
        print(log_string)

    def log_adapt(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, self.current_learning_iteration)
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))
        
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, self.current_learning_iteration)
        self.writer.add_scalar('Perf/total_fps', fps, self.current_learning_iteration)
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], self.current_learning_iteration)
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], self.current_learning_iteration)
        self.writer.add_scalar('Perf/gpu_allocated', torch.cuda.memory_allocated(self.device) / 1024 ** 3, self.current_learning_iteration)
        self.writer.add_scalar('Perf/gpu_occupied', torch.cuda.mem_get_info(self.device)[1] / 1024 ** 3, self.current_learning_iteration)
        self.writer.add_scalar('Train/mean_reward_each_timestep', statistics.mean(locs['rframebuffer']), self.current_learning_iteration)
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), self.current_learning_iteration)
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), self.current_learning_iteration)
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {self.current_learning_iteration}/{locs['tot_iter']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
                        )
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs["losses"]['value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs["losses"]['surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
                        )

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (self.current_learning_iteration + 1 - locs["start_iter"]) * (
                               locs['tot_iter'] - self.current_learning_iteration):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        run_state_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }
        if hasattr(self.alg, "lr_scheduler"):
            run_state_dict["lr_scheduler_state_dict"] = self.alg.lr_scheduler.state_dict()
        torch.save(run_state_dict, path)

    def save_base_actor(self, path, infos=None):
        run_state_dict = {
            'model_state_dict': self.alg.actor_critic.actor.actor.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }
        if hasattr(self.alg, "lr_scheduler"):
            run_state_dict["lr_scheduler_state_dict"] = self.alg.lr_scheduler.state_dict()
        torch.save(run_state_dict, path)

    def save_env_factor_encoder(self, path, infos=None):
        run_state_dict = {
            'model_state_dict': self.alg.actor_critic.actor.env_factor_encoder.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }
        if hasattr(self.alg, "lr_scheduler"):
            run_state_dict["lr_scheduler_state_dict"] = self.alg.lr_scheduler.state_dict()
        torch.save(run_state_dict, path)

    
    def save_adapt_module_encoder(self, path, infos=None):
        run_state_dict = {
            'model_state_dict': self.adaptation_encoder.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }
        if hasattr(self.alg, "lr_scheduler"):
            run_state_dict["lr_scheduler_state_dict"] = self.alg.lr_scheduler.state_dict()
        torch.save(run_state_dict, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer and "optimizer_state_dict" in loaded_dict:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        if "lr_scheduler_state_dict" in loaded_dict:
            if not hasattr(self.alg, "lr_scheduler"):
                print("Warning: lr_scheduler_state_dict found in checkpoint but no lr_scheduler in algorithm. Ignoring.")
            else:
                self.alg.lr_scheduler.load_state_dict(loaded_dict["lr_scheduler_state_dict"])
        elif hasattr(self.alg, "lr_scheduler"):
            print("Warning: lr_scheduler_state_dict not found in checkpoint but lr_scheduler in algorithm. Ignoring.")
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']


    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
