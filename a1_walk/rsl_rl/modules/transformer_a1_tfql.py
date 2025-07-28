import torch

import torch.nn as nn


from rsl_rl.utils.mappings import Mapping

class ActorObsTokenizer(nn.Module):
    def __init__(self, *, dim, num_actions, nbodies=14):
        super().__init__()
        self.activation = nn.ELU()
        self.num_actions = num_actions
        self.nb = nbodies
     

    def forward(self, x, latent_vector=None):
        # print("latent_vector", latent_vector)
        x = torch.cat((x[:, 0:48], latent_vector, x[:, -1].unsqueeze(1)), dim=1)
        
        return x


class CriticObsTokenizer(nn.Module):
    def __init__(self, *, dim, num_actions,nbodies = 14):
        super().__init__()
        self.activation = nn.ELU()

    def forward(self, x):
        
        return x


    



class Env_Factor_Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Env_Factor_Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, mask=None):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class BodyActor(nn.Module):
    def __init__(self, name, net, embedding_dim, action_dim, stack_time=True, global_input=False, mu_activation=None, device='cuda',nbodies = 15):
        super(BodyActor, self).__init__()
        self.global_input = global_input
        # self.adaption_encoder = MLPClass(input_dim = 187+17, output_dim = 120)
        self.tokenizer = ActorObsTokenizer(dim = embedding_dim, num_actions=action_dim, nbodies = nbodies)
        self.net = net
        self.num_actions = action_dim
        self.tokenizer.to(device)
        self.net.to(device)

        self.mu_activation = mu_activation

    def forward(self, x, latent_vector):
        # print("input actor:", x.shape)
        # latent_vector = self.adaption_encoder(x[:,(12+3*self.num_actions):(12+3*self.num_actions+187+17)])
        x = self.tokenizer(x, latent_vector = latent_vector) # (B, nbodies, lookback_steps, embedding_dim)
      
        x = self.net(x)

        if self.mu_activation is not None:
            x = self.mu_activation(x)
        
        return x

    def mode(self, x):
        return self.forward(x)

class TFQL_BodyActor(nn.Module):
    def __init__(self, name, net, embedding_dim, action_dim, stack_time=True, global_input=False, mu_activation=None, device='cuda',nbodies = 15):
        super(TFQL_BodyActor, self).__init__()

        self.env_factor_encoder = Env_Factor_Encoder(input_dim=187+17, output_dim=8)
        self.adapt_encoder = Adaptation_Encoder(input_dim=48, output_dim=8, hidden_dim=64, tsteps=50)
        
        self.alpha = 0

        # 使用独立的 BodyActor 类
        self.actor = BodyActor(name, net, embedding_dim, action_dim, global_input=global_input, mu_activation=mu_activation, device=device, nbodies=nbodies)

        self.num_actions = action_dim


    def forward(self, x, sa_pair):
        # 生成 latent_vector
        env_factor_latent = self.env_factor_encoder(x[:,(12+3*self.num_actions):(12+3*self.num_actions+187+17)])
        # print("env_factor_latent:", env_factor_latent.shape)
        # print("sa_pair:", sa_pair.shape, sa_pair)
        adapt_encoder_latent = self.adapt_encoder(sa_pair)
        # print("adapt_encoder_latent:", adapt_encoder_latent.shape)
        # exit(0)
        # print("alpha:", self.alpha)
        latent_vector = (1-self.alpha)*env_factor_latent + self.alpha * adapt_encoder_latent
        
        # 使用生成的 latent_vector 和输入 x 进行后续处理
        return self.actor(x, latent_vector), env_factor_latent, adapt_encoder_latent


class Adaptation_Encoder(nn.Module):
    def __init__(self, input_dim=48, output_dim=120, hidden_dim=64, tsteps=50):
        super(Adaptation_Encoder, self).__init__()

        # MLP层：输入最近的状态和动作，并映射到32维表示
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 输入 -> 隐藏层
            nn.ReLU(),  # 激活函数
            nn.Linear(hidden_dim, 32)  # 隐藏层 -> 输出层（32维）
        )

        # 1D卷积层：捕获时间维度的时序相关性
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8, stride=4),  # 第一层卷积
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1),  # 第二层卷积
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1)  # 第三层卷积
        )

        # 线性层，用于估计 zˆt
        self.fc = nn.Linear(32*3, output_dim)  # 输入 32 * 3，输出维度为 output_dim

        self.tsteps = tsteps

    def forward(self, x):
        # 先通过MLP层
        bs = x.shape[1]
        # print("bs:",bs)
        # print("self.tsteps:",self.tsteps)
        projection = self.mlp(x.reshape([bs * self.tsteps, -1]))  # 将输入展平为 [batch_size * tsteps, input_dim]
        
        # 重新调整为卷积层所需的形状： (B, 32, T)
        # print("proj1:",projection.shape)
        projection = projection.view(bs, 32, self.tsteps)
        # print("proj:",projection.shape)
        # 通过卷积层
        x = self.cnn(projection)

        # Flatten后通过线性层得到最终输出
        x = x.view(x.size(0), -1)  # Flatten: (B, C * L) -> (B, flattened_size)
        # print(x.shape)
        z_hat = self.fc(x)  # 线性层估计 zˆt

        return z_hat


class BodyCritic(nn.Module):
    def __init__(self, mapping, net, action_dim, embedding_dim, stack_time=True, global_input=False, device='cuda', nbodies = 14):
        super(BodyCritic, self).__init__()
        self.global_input = global_input
        # self.tokenizer = ObsTokenizer(mapping, embedding_dim, stack_time, device)
        self.tokenizer = CriticObsTokenizer(dim = embedding_dim, num_actions=action_dim, nbodies = nbodies)
        self.net = net


        self.tokenizer.to(device)
        self.net.to(device)


    def forward(self, x):
        # print("input critic:", x.shape)
        x  = self.tokenizer(x) # (B, nbodies, lookback_steps, embedding_dim)

        x = self.net(x)
        
        
        return x

    def mode(self, x):
        return self.forward(x)
    

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=(256, 128), activation=nn.ReLU):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation  # 传入的是类，而不是实例
        
        layers = []
        in_size = input_dim
        for next_size in hidden_sizes:
            layers.append(nn.Linear(in_size, next_size))
            layers.append(activation())  # 实例化激活函数
            in_size = next_size
        layers.append(nn.Linear(in_size, output_dim))  # 修改输出维度
        
        self.model = nn.Sequential(*layers)
        self.init_weights()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.model(x)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def init_weights(self):
        self.apply(self._init_weights)


class Transformer(torch.nn.Module):
    def __init__(self, input_dim, dim_feedforward=128, nhead=6, nlayers=3, mask_position=None, numbodies = 14):
        super(Transformer, self).__init__()
        self.mask_position = mask_position
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=0.)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        max_nbodies = numbodies  # TODO: not hard-coded
        self.embed_absolute_position = nn.Embedding(max_nbodies, embedding_dim=input_dim)  # max_num_limbs

        self.output_dim = input_dim
        self.init_weights()

    def forward(self, x, mask = None):
        batch_size, nbodies, embedding_dim = x.shape
        
        limb_indices = torch.arange(0, nbodies, device=x.device)
        limb_idx_embedding = self.embed_absolute_position(limb_indices)
        x = x + limb_idx_embedding
        
        if mask is not None:
            # Pass the masked x to the Transformer encoder
            x = self.encoder(x, src_key_padding_mask=mask)
        else:
            x = self.encoder(x)
        return x

    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def init_weights(self):
        self.apply(self._init_weights)
    
