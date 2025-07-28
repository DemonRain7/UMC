import torch

import torch.nn as nn


from rsl_rl.utils.mappings import Mapping

class ActorObsTokenizer(nn.Module):
    def __init__(self, dim=120):
        super().__init__()
        self.activation = nn.ELU()
        
        # 定义MLP层
        self.mlp = nn.Sequential(
            nn.Linear(48 + 12, 64),
            nn.ELU(),
            nn.Linear(64, dim)
        )
    
    def forward(self, x):
        # 选取输入并通过MLP
        x_selected = torch.cat((x[:, 0:48], x[:, -12:]), dim=1)
        x_out = self.mlp(x_selected)
        
        return x_out.unsqueeze(1) 


class CriticObsTokenizer(nn.Module):
    def __init__(self, *, dim, num_actions,nbodies = 14):
        super().__init__()
        self.activation = nn.ELU()
        self.num_actions = num_actions
        self.nb = nbodies - 1 # 无法检测到机器人内部损坏

        
        self.linear_embedding_selfobs = nn.ModuleList(
            [nn.Linear(3, dim) for _ in range(4)]
        )
        # 定义不同类型输入的linear embedding
        self.linear_embedding_root = nn.Linear(12, dim)
        # self.linear_embedding_heights = nn.Linear(187, dim)

    def forward(self, x):
        # max_value = torch.max(x)
        # print("max:", max_value)
        batch_size, nbodies = x.shape[0], self.nb   # TODO:假设 nbodies 是 14，需根据实际情况调整
        # Process x with linear embeddings
        x_root = self.linear_embedding_root(x[:, 0:12])
        # x_heights = self.linear_embedding_heights(x[:, 48:235])

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
            elif 3 <=i <6:
                x_node = self.linear_embedding_selfobs[1](node_input)  # 使用不同的线性层
            elif 6 <=i <9:
                x_node = self.linear_embedding_selfobs[2](node_input)  # 使用不同的线性层
            elif 9 <=i <12:
                x_node = self.linear_embedding_selfobs[3](node_input)  # 使用不同的线性层
            x_nodes.append(x_node)

        x_nodes = torch.stack(x_nodes, dim=1)
        x = torch.cat((x_nodes, x_root.unsqueeze(1)), dim=1)
        
        return x


class ActionDetokenizer(nn.Module):
    def __init__(self, embedding_dim, action_dim, global_input=False, device='cuda'):
        super(ActionDetokenizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.device = device

        # 2-layer LSTM
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=2, batch_first=True)

        # Linear layer to map LSTM output to action_dim
        self.output_layer = nn.Linear(embedding_dim, action_dim)

    def forward(self, x):
        """
        x: class_token from Transformer, shape (batch_size, 1, embedding_dim)
        """
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)  # (batch_size, 1, embedding_dim)
        hidden_state = lstm_out[:, -1, :]  # Take the last timestep output

        # Map to action_dim
        action = self.output_layer(hidden_state)  # (batch_size, action_dim)

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

class BodyActor(nn.Module):
    def __init__(self, name, net, embedding_dim, action_dim, stack_time=True, global_input=False, mu_activation=None, device='cuda',nbodies = 14):
        super(BodyActor, self).__init__()
        self.global_input = global_input
        # self.tokenizer = ObsTokenizer(name, embedding_dim, stack_time, device)
        self.tokenizer = ActorObsTokenizer(dim = embedding_dim)
        self.net = net
        self.detokenizer = ActionDetokenizer(net.output_dim, action_dim, global_input, device=device)

        self.tokenizer.to(device)
        self.net.to(device)
        self.detokenizer.to(device)

        self.mu_activation = mu_activation

    def forward(self, x):
        # print("input actor:", x.shape)
        x = self.tokenizer(x) # (B, nbodies, lookback_steps, embedding_dim)

        x = self.net(x)

        x = self.detokenizer(x)

        if self.mu_activation is not None:
            x = self.mu_activation(x)
        
        return x

    def mode(self, x):
        return self.forward(x)

class BodyCritic(nn.Module):
    def __init__(self, mapping, net, action_dim, embedding_dim, stack_time=True, global_input=False, device='cuda', nbodies = 13):
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


class Transformer(nn.Module):
    def __init__(self, input_dim, dim_feedforward=256, nhead=16, nlayers=2, mask_position=None, numbodies=1):
        super(Transformer, self).__init__()
        self.mask_position = mask_position
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=0.)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        max_nbodies = numbodies  # TODO: not hard-coded

        self.class_token = nn.Parameter(torch.zeros(1, 1, input_dim))  # 可学习的 class token
        self.output_dim = input_dim
        self.init_weights()

    def forward(self, x):
        batch_size, nbodies, embedding_dim = x.shape
        
        # 生成 class token，并复制 batch_size 份
        class_token = self.class_token.expand(batch_size, -1, -1)
        
        # 将 class token 拼接到 x 的最前面
        x = torch.cat([class_token, x], dim=1)
    
        x = self.encoder(x)
        
        # 仅返回 class token
        return x[:, 0:1, :]
    
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
    
class TFQL_BodyActor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("TFQL_BodyActor should not be used.")
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("TFQL_BodyActor forward() should not be used.")


class Transformer_SMS(torch.nn.Module):
    def __init__(self, input_dim, dim_feedforward=128, nhead=6, nlayers=3, mask_position=None, numbodies = 14):
        super(Transformer_SMS, self).__init__()
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
    
