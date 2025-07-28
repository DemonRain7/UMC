import torch

import torch.nn as nn


from rsl_rl.utils.mappings import Mapping

class ActorObsTokenizer(nn.Module):
    def __init__(self, *, dim, num_actions, nbodies=14):
        super().__init__()
        self.activation = nn.ELU()
        self.num_actions = num_actions
        self.nb = nbodies
       
        # 定义不同类型输入的linear embedding
        self.linear_embedding_selfobs = nn.ModuleList(
            [nn.Linear(3, dim) for _ in range(6)]
        )
        # 定义不同类型输入的linear embedding
        self.linear_embedding_root = nn.Linear(12, dim)
        # self.linear_embedding_heights = nn.Linear(187, dim)
        self.linear_embedding_isdoflimit = nn.Linear(3, dim)

    def forward(self, x, latent_vector=None):
        # max_value = torch.max(x)
        # print("max:", max_value)
        # print(x.shape)
        batch_size, nbodies = x.shape[0], self.nb 
        # Initialize mask
        # mask = None
        mask = torch.zeros((batch_size, nbodies), dtype=torch.bool, device=x.device)

        # # TODO:NOT HARD-CODED
        # # Convert x to a boolean tensor where True represents values equal to 1 (masked)
        bool_x = (x[:, -self.num_actions:] == 1)  # Check last 12 bits of x
        # # print(bool_x)

        # # Assign mask for each environment based on DOF mask bits
        mask[:, :self.num_actions] = bool_x

        # print("mask:",mask)
        # print(self.num_actions)

        # Process x with linear embeddings
        x_root = self.linear_embedding_root(x[:, 0:12])
        # x_heights = self.linear_embedding_heights(x[:,  (12+3*self.num_actions):(12+3*self.num_actions+187)])
        x_isdoflimit = self.linear_embedding_isdoflimit(x[:, -(self.num_actions+3):-self.num_actions])
        # print(latent_vector.shape)
        # 对每个node做embedding
        x_nodes = []
        # print(self.num_actions)
        for i in range(self.num_actions):
           
            node_input = torch.cat((
                x[:, 12 + i].unsqueeze(1),
                x[:, 12 + self.num_actions + i].unsqueeze(1),
                x[:, 12 + 2 * self.num_actions + i].unsqueeze(1)
            ), dim=1)
            # print(i,node_input)
            if 0 <=i <3:
                x_node = self.linear_embedding_selfobs[0](node_input)  # 使用不同的线性层
            elif i==3:
                x_node = self.linear_embedding_selfobs[1](node_input)  # 使用不同的线性层
            elif 4 <= i <6:
                x_node = self.linear_embedding_selfobs[2](node_input)  # 使用不同的线性层
            elif 6 <=i <9:
                x_node = self.linear_embedding_selfobs[3](node_input)  # 使用不同的线性层
            elif i==9:
                x_node = self.linear_embedding_selfobs[4](node_input)  # 使用不同的线性层
            elif 10 <= i < 12:
                x_node = self.linear_embedding_selfobs[5](node_input)  # 使用不同的线性层
            x_nodes.append(x_node)
            
        x_nodes = torch.stack(x_nodes, dim=1)
        x = torch.cat((x_nodes, x_root.unsqueeze(1), latent_vector.unsqueeze(1), x_isdoflimit.unsqueeze(1)), dim=1)
        
        return x, mask


class CriticObsTokenizer(nn.Module):
    def __init__(self, *, dim, num_actions,nbodies = 14):
        super().__init__()
        self.activation = nn.ELU()
        self.num_actions = num_actions
        self.nb = nbodies - 1 # 无法检测到机器人内部损坏

        
        self.linear_embedding_selfobs = nn.ModuleList(
            [nn.Linear(3, dim) for _ in range(6)]
        )
        # 定义不同类型输入的linear embedding
        self.linear_embedding_root = nn.Linear(12, dim)
        self.linear_embedding_pri_obs = nn.Linear(187+17, dim)

    def forward(self, x):
        # max_value = torch.max(x)
        # print("max:", max_value)
        batch_size, nbodies = x.shape[0], self.nb   # TODO:假设 nbodies 是 14，需根据实际情况调整
        # Process x with linear embeddings
        x_root = self.linear_embedding_root(x[:, 0:12])
        x_pri_obs = self.linear_embedding_pri_obs(x[:, (12+3*self.num_actions):(12+3*self.num_actions+187+17)])

        # 对每个node做embedding
        x_nodes = []
        for i in range(self.num_actions):
            node_input = torch.cat((
                x[:, 12 + i].unsqueeze(1),
                x[:, 12 + self.num_actions + i].unsqueeze(1),
                x[:, 12 + 2 * self.num_actions + i].unsqueeze(1)
            ), dim=1)
            if 0 <=i <3:
                x_node = self.linear_embedding_selfobs[0](node_input)  # 使用不同的线性层
            elif i==3:
                x_node = self.linear_embedding_selfobs[1](node_input)  # 使用不同的线性层
            elif 4 <= i <6:
                x_node = self.linear_embedding_selfobs[2](node_input)  # 使用不同的线性层
            elif 6 <=i <9:
                x_node = self.linear_embedding_selfobs[3](node_input)  # 使用不同的线性层
            elif i==9:
                x_node = self.linear_embedding_selfobs[4](node_input)  # 使用不同的线性层
            elif 10 <= i < 12:
                x_node = self.linear_embedding_selfobs[5](node_input)  # 使用不同的线性层
            x_nodes.append(x_node)

        x_nodes = torch.stack(x_nodes, dim=1)
        x = torch.cat((x_nodes, x_root.unsqueeze(1), x_pri_obs.unsqueeze(1)), dim=1)
        
        return x

    
class ActionDetokenizer(torch.nn.Module):
    def __init__(self, name, embedding_dim, action_dim, global_input=False, device='cuda'):
        super(ActionDetokenizer, self).__init__()
        # print(name)
        self.mapping = Mapping(name)
        self.map = self.mapping.get_map()
        # print(self.map)
        self.nbodies = len(self.map.keys())

        self.embedding_dim = embedding_dim
        self.action_dim = action_dim

        self.device = device

        base = lambda output_dim : torch.nn.Linear(embedding_dim, output_dim)
        self.detokenizers = torch.nn.ModuleDict()
        # if global_input:
        #     self.detokenizers['global'] = base(action_dim)
        # else:
        #     for k in self.map.keys():
        #         self.detokenizers[k] = base(len(self.map[k][1])) 
        for k in self.map.keys():
            self.detokenizers[k] = base(len(self.map[k][1])) 

    def forward(self, x):

        # if 'global' in self.detokenizers:
        #     return self.detokenizers['global'](x.to(self.device))
        
        action = torch.zeros(x.shape[0], self.action_dim).to(self.device)
        for i, k in enumerate(self.map.keys()):
            curr_action = self.detokenizers[k](x[:,i,:])
            # # 打印当前 detokenizer 的映射
            # print(f"Mapping key: {k},{i}")
            # print(f"Detokenizer layer for {k}: {self.detokenizers[k]}")
            # print(f"DOF mapping for {k}: {self.map[k][0]}")  # 打印DOF
            # print(f"DOF indices for {k}: {self.map[k][1]}")  # 打印DOF的index
            action[:, self.map[k][1]] = curr_action.float()
        return action
    
class ValueDetokenizer(torch.nn.Module):
    def __init__(self, name, embedding_dim, global_input=False, device='cuda'):
        super(ValueDetokenizer, self).__init__()

        self.mapping = Mapping(name)
        self.map = self.mapping.get_map()

        self.nbodies = len(self.map.keys())

        self.embedding_dim = embedding_dim

        self.device = device

        base = lambda output_dim : torch.nn.Linear(embedding_dim, output_dim)
        self.detokenizers = torch.nn.ModuleDict()
        # if global_input:
        #     self.detokenizers['global'] = base(1)
        # else:
        #     for k in self.map.keys():
        #         self.detokenizers[k] = base(1) 
        for k in self.map.keys():
                self.detokenizers[k] = base(1) 

    def forward(self, x):

        # if 'global' in self.detokenizers:
        #     return self.detokenizers['global'](x.to(self.device))
        
        values = torch.zeros(x.shape[0], x.shape[1]).to(self.device)
        for i, k in enumerate(self.map.keys()):
            values[:,i] = self.detokenizers[k](x[:,i,:]).squeeze(-1)
        return torch.mean(values, dim=1, keepdim=True)


class Env_Factor_Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Env_Factor_Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, mask=None):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class BodyActor(nn.Module):
    def __init__(self, name, net, embedding_dim, action_dim, stack_time=True, global_input=False, mu_activation=None, device='cuda',nbodies = 15):
        super(BodyActor, self).__init__()
        self.global_input = global_input
        # self.adaption_encoder = MLPClass(input_dim = 187+17, output_dim = 120)
        self.tokenizer = ActorObsTokenizer(dim = embedding_dim, num_actions=action_dim, nbodies = nbodies)
        self.net = net
        self.detokenizer = ActionDetokenizer(name, net.output_dim, action_dim, global_input, device=device)
        self.num_actions = action_dim
        self.tokenizer.to(device)
        self.net.to(device)
        self.detokenizer.to(device)

        self.mu_activation = mu_activation

    def forward(self, x, latent_vector):
        # print("input actor:", x.shape)
        # latent_vector = self.adaption_encoder(x[:,(12+3*self.num_actions):(12+3*self.num_actions+187+17)])
        x, mask = self.tokenizer(x, latent_vector = latent_vector) # (B, nbodies, lookback_steps, embedding_dim)
        # 判断 mask 是否全为 False
        mask = None
        # mask = None
        # print(mask)
        if self.global_input:
            x = self.net(x)
        else :
            x = self.net(x, mask= mask)

        x = self.detokenizer(x)

        if self.mu_activation is not None:
            x = self.mu_activation(x)
        
        return x

    def mode(self, x):
        return self.forward(x)

class Env_Factor_BodyActor(nn.Module):
    def __init__(self, name, net, embedding_dim, action_dim, stack_time=True, global_input=False, mu_activation=None, device='cuda',nbodies = 15):
        super(Env_Factor_BodyActor, self).__init__()

        self.env_factor_encoder = Env_Factor_Encoder(input_dim=187+17, output_dim=120)
        
        # 使用独立的 BodyActor 类
        self.actor = BodyActor(name, net, embedding_dim, action_dim, global_input=global_input, mu_activation=mu_activation, device=device, nbodies=nbodies)

        self.num_actions = action_dim

    def forward(self, x):
        # 生成 latent_vector
        latent_vector = self.env_factor_encoder(x[:,(12+3*self.num_actions):(12+3*self.num_actions+187+17)])
        
        # 使用生成的 latent_vector 和输入 x 进行后续处理
        return self.actor(x, latent_vector)


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
        self.detokenizer = ValueDetokenizer(mapping, net.output_dim, global_input, device=device)

        self.tokenizer.to(device)
        self.net.to(device)
        self.detokenizer.to(device)

    def forward(self, x):
        # print("input critic:", x.shape)
        x  = self.tokenizer(x) # (B, nbodies, lookback_steps, embedding_dim)

        x = self.net(x)
        
        x = self.detokenizer(x)
        
        return x

    def mode(self, x):
        return self.forward(x)
    
class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_sizes=(1024, 1024), activation=torch.nn.ReLU()):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation

        self.layers = []
        in_size = input_dim
        for next_size in hidden_sizes:
            self.layers.append(torch.nn.Linear(in_size, next_size))
            self.layers.append(self.activation)
            in_size = next_size
        self.layers.append(torch.nn.Linear(in_size, input_dim))
        
        self.model = torch.nn.Sequential(*self.layers)

        self.output_dim = input_dim
        self.init_weights()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.model(x)
    
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
        # print(x.shape)
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
    
