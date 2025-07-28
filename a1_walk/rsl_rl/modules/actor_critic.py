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

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
import torch.nn.functional as F

from rsl_rl.modules.transformers import  Transformer,  BodyActor, BodyCritic, Transformer_SMS
from rsl_rl.modules.transformers import TFQL_BodyActor, MLP
from rsl_rl.modules.transformer_a1_rld import Env_Factor_BodyActor
from rsl_rl.modules.transformer_bt import BodyTransformer_Actor, BodyTransformer_Critic, BodyActor_Bodytrf, BodyCritic_Bodytrf

class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actions,
                        env_name,
                        nbodies,
                        num_actor_obs,
                        num_critic_obs,
                        actor_hidden_dims=[150,150,150],
                        critic_hidden_dims=[150,150,150],
                        mask_pos = None,
                        actor_type='mlp',
                        critic_type='mlp',
                        embedding_dim=64,
                        nheads=2,
                        nlayers=10,
                        dim_feedforward=256,
                        is_mixed=False,
                        init_noise_std=1.0,
                        device='cuda',
                        mu_activation=None,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()


        print("params")
        print(f"num_actions: {num_actions}")
        print(f"env_name: {env_name}")
        print(f"nbodies: {nbodies}")
        print(f"actor_type: {actor_type}")
        print(f"critic_type: {critic_type}")
        print(f"embedding_dim: {embedding_dim}")
        print(f"nheads: {nheads}")
        print(f"nlayers: {nlayers}")
        print(f"dim_feedforward: {dim_feedforward}")
        print(f"is_mixed: {is_mixed}")
        print(f"init_noise_std: {init_noise_std}")
        print(f"device: {device}")
        print(f"mu_activation: {mu_activation}")
        print(f"mask_pos: {mask_pos}")

        activation_org = "elu"
        activation = get_activation(activation_org)
        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        
        if actor_type == 'mlp':
            print(f"mlp_input_dim_a: {num_actor_obs}")
            # Actor Function
            actor_layers = []
            # actor_layers.append(PrintLayer())
            actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
            actor_layers.append(activation)
            for l in range(len(actor_hidden_dims)):
                if l == len(actor_hidden_dims) - 1:
                    actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                else:
                    actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                    actor_layers.append(activation)
            self.actor = nn.Sequential(*actor_layers)

        elif actor_type == 'transformer':
            actor_net = Transformer(embedding_dim, dim_feedforward=dim_feedforward, nhead=nheads, nlayers=nlayers, mask_position= mask_pos,numbodies = nbodies)
            self.actor = BodyActor(env_name, actor_net, embedding_dim=embedding_dim, action_dim=num_actions, stack_time=False, global_input=actor_type=='mlp', mu_activation=get_activation(mu_activation), device=device,nbodies = nbodies)
        elif actor_type == 'transformer_sms':
            actor_net = Transformer(embedding_dim, dim_feedforward=dim_feedforward, nhead=4, nlayers=2, mask_position= mask_pos,numbodies = nbodies)
            self.actor = BodyActor(env_name, actor_net, embedding_dim=embedding_dim, action_dim=num_actions, stack_time=False, global_input=actor_type=='mlp', mu_activation=get_activation(mu_activation), device=device,nbodies = nbodies)

        elif actor_type == 'body_transformer':
            actor_net = BodyTransformer_Actor(env_name, embedding_dim, dim_feedforward=dim_feedforward, nhead=nheads, num_layers=nlayers, is_mixed=is_mixed, first_hard_layer=0)
            self.actor = BodyActor_Bodytrf(env_name, actor_net, embedding_dim=embedding_dim, action_dim=num_actions, stack_time=False, global_input=actor_type=='mlp', mu_activation=get_activation(mu_activation), device=device,nbodies = nbodies)
        elif actor_type == 'transformer_rma':
            actor_net = Transformer(embedding_dim, dim_feedforward=dim_feedforward, nhead=nheads, nlayers=nlayers, mask_position= mask_pos,numbodies = nbodies)
            self.actor = Env_Factor_BodyActor(env_name, actor_net, embedding_dim=embedding_dim, action_dim=num_actions, stack_time=False, global_input=actor_type=='mlp', mu_activation=get_activation(mu_activation), device=device,nbodies = nbodies)
        elif actor_type == 'mlp_tfql':
            actor_net = MLP(input_dim=48+8+12, output_dim=12, hidden_sizes=[256, 128], activation=torch.nn.ReLU)
            self.actor = TFQL_BodyActor(env_name, actor_net, embedding_dim=embedding_dim, action_dim=num_actions, stack_time=False, global_input=actor_type=='mlp', mu_activation=get_activation(mu_activation), device=device,nbodies = nbodies)
        
        elif actor_type == 'transformer_rma_phase2':
            print("No Need to Construct Actor!")
        else:
            raise ValueError("invalid actor type")
            return
        
        
        if critic_type == 'mlp':
            print(f"mlp_input_dim_c: {num_critic_obs}")
             # Value function
            critic_layers = []
            # critic_layers.append(PrintLayer())
            critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
            critic_layers.append(activation)
            for l in range(len(critic_hidden_dims)):
                if l == len(critic_hidden_dims) - 1:
                    critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
                else:
                    critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                    critic_layers.append(activation)
            self.critic = nn.Sequential(*critic_layers)
        elif critic_type == 'transformer':
            critic_net = Transformer(embedding_dim, dim_feedforward=dim_feedforward, nhead=nheads, nlayers=nlayers, mask_position= mask_pos,numbodies = nbodies)
            self.critic = BodyCritic(env_name, critic_net, action_dim=num_actions, embedding_dim=embedding_dim, stack_time=False, global_input=critic_type=='mlp', device=device, nbodies = nbodies)
        elif critic_type == 'transformer_sms':
            critic_net = Transformer_SMS(embedding_dim, dim_feedforward=dim_feedforward, nhead=nheads, nlayers=nlayers, mask_position= mask_pos,numbodies = nbodies)
            self.critic = BodyCritic(env_name, critic_net, action_dim=num_actions, embedding_dim=embedding_dim, stack_time=False, global_input=critic_type=='mlp', device=device, nbodies = nbodies)
        
        elif critic_type == 'body_transformer':
            critic_net = BodyTransformer_Critic(env_name, embedding_dim, dim_feedforward=dim_feedforward, nhead=nheads, num_layers=nlayers, is_mixed=is_mixed, first_hard_layer=0)
            self.critic = BodyCritic_Bodytrf(env_name, critic_net,  embedding_dim=embedding_dim, stack_time=False, global_input=critic_type=='mlp', device=device)
        elif actor_type == 'mlp_tfql':
            critic_net = MLP(input_dim=252, output_dim=1, hidden_sizes=[256, 128], activation=torch.nn.ReLU)
            self.critic = BodyCritic(env_name, critic_net, action_dim=num_actions, embedding_dim=embedding_dim, stack_time=False, global_input=critic_type=='mlp', device=device, nbodies = nbodies)
        elif critic_type == 'transformer_rma_phase2':
            print("No Need to Construct Critic!")
        else:
            raise ValueError("invalid critic type")
            return
       
        if critic_type != 'transformer_rma_phase2' and actor_type != 'transformer_rma_phase2':
            print(f"Actor Model: {self.actor}")
            print(f"Critic Model: {self.critic}")
        
            actor_nparams = sum(p.numel() for p in self.actor.parameters() if p.requires_grad)
            critic_nparams = sum(p.numel() for p in self.critic.parameters() if p.requires_grad)

            print('actor params:', actor_nparams)
            print('critic params:', critic_nparams)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        self.env_factor_latent = None
        self.adapt_encoder_latent = None
        

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def update_distribution_tfql(self, observations, sa_pair):
        mean, self.env_factor_latent, self.adapt_encoder_latent = self.actor(observations, sa_pair)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act_tfql(self, observations, sa_pair, **kwargs):
        self.update_distribution_tfql(observations, sa_pair)
        return self.distribution.sample()

    def act_inference_tfql(self, observations, sa_pair):
        actions_mean, _ , _ = self.actor(observations, sa_pair)
        return actions_mean

    def compute_adapt_loss(self):
        if self.env_factor_latent is not None and self.adapt_encoder_latent is not None:
            return F.mse_loss(self.env_factor_latent, self.adapt_encoder_latent)
        else:
            raise ValueError("self.env_factor_latent or self.adapt_encoder_latent is None!")

    @torch.no_grad()
    def clip_std(self, min= None, max= None):
        self.std.copy_(self.std.clip(min= min, max= max))

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None