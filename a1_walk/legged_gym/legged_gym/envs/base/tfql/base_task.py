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

import sys
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import torch
import random

# Base class for RL tasks
class BaseTask():

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym()

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless
        self.curr_iter = 0
        self.dof_limit_state = []
        self.dof_limit_dict = {}


        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        if self.headless == True and self.cfg.sim.no_camera:
            self.graphics_device_id = -1

        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else: 
            self.privileged_obs_buf = None
            # self.num_privileged_obs = self.num_obs

        self.extras = {}

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None
        
        # For TFQL
        # 生成若干malfunc_time
        self.malfunc_time = [random.randint(75, 125) for _ in range(self.num_envs)]
        self.dmg_flag =  torch.zeros(self.num_envs, 12, device=self.device, requires_grad=False)

        # 暂存当前随机数生成器的状态
        saved_state = random.getstate()
        random.seed(42)
        for i in range(self.num_envs):
            # 生成1个随机dof损坏
            # 从0到11中随机选取
            faulty_dof_indices = random.sample(range(12), 1)  # 从0到11中不重复地选择num_faulty_dofs个元素
            # print(faulty_dof_indices)
            self.dof_limit_state.append((i, faulty_dof_indices))

         # 在 create_env 函数执行完后，将 dof_limit_state 转为字典
        self.dof_limit_dict = {env_id: faulty_dof_indices for (env_id, faulty_dof_indices) in self.dof_limit_state}
        print("dof_limit_dict:", self.dof_limit_dict)
        random.setstate(saved_state)

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

    def get_observations(self):
        return self.obs_buf
    
    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False),self.curr_iter)
        return obs, privileged_obs

    def step(self, actions, tmp_iter):
        self.curr_iter = tmp_iter
        raise NotImplementedError

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)