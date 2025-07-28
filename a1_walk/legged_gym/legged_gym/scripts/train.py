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

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

from legged_gym.debugger import break_into_debugger


import sys
import contextlib

class Logger(object):
    def __init__(self, filename="log.txt", log_dir="./logs"):
        # 创建日志目录（如果不存在）
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 设置日志文件路径
        self.log_path = os.path.join(log_dir, filename)
        self.terminal = sys.stdout
        self.log = open(self.log_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg, log_dir = task_registry.make_alg_runner(env=env, name=args.task, args=args, env_cfg=env_cfg)
     # 重定向标准输出到日志文件
    log_filename = "training_output.log"
    log_directory = log_dir
    sys.stdout = Logger(log_filename, log_directory)
    if train_cfg.policy.actor_type == "transformer_rma_phase2":
        ppo_runner.learn_adaption_encoder(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    elif train_cfg.policy.actor_type == "mlp_tfql":
        ppo_runner.learn_tfql(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    else:
        ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
    sys.stdout.log.close()
