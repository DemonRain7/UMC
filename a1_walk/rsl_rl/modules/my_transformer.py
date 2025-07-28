from einops import rearrange, repeat
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


from einops import rearrange, repeat
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)



class Trf_Policy_Actor(nn.Module):
    def __init__(self, *,  input_dim, num_classes, dim, depth, heads, mlp_dim, activation, pool = 'cls', dim_head = 64):
        super().__init__()
        self.input_dim = input_dim  
        self.d_model = dim
        self.num_actions = num_classes
        self.activation = get_activation(activation)

        # # 定义不同类型输入的linear embedding
        # self.linear_embedding_actions = nn.Sequential(
        #     nn.Linear(self.num_actions, dim),
        #     self.activation,
        #     nn.LayerNorm(dim)
        # )
        # self.linear_embedding_root = nn.Sequential(
        #     nn.Linear(9, dim),
        #     self.activation,
        #     nn.LayerNorm(dim)
        # )
        # self.linear_embedding_commands = nn.Sequential(
        #     nn.Linear(3, dim),
        #     self.activation,
        #     nn.LayerNorm(dim)
        # )
        # self.linear_embedding_selfobs = nn.Sequential(
        #     nn.Linear(3, dim),  # 对于每个节点
        #     self.activation,
        #     nn.LayerNorm(dim)
        # )

          # 定义不同类型输入的linear embedding

        self.linear_embedding_root = nn.Linear(12, dim)
        self.linear_embedding_selfobs= nn.Linear(3, dim)
    
        
        # 位置编码
        max_seq_len = 1 + self.num_actions  # 3: actions, root, command; + self.num_actions for nodes


        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, dim))
        # self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim=dim)  # max_num_limbs

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        # encoder_layer =  nn.TransformerEncoderLayer(d_model=self.d_model, nhead=heads, dim_feedforward=mlp_dim, batch_first=True, dropout=0.)
        # self.transformer = nn.TransformerEncoder(encoder_layer,num_layers=depth)

        # self.pool = pool
        # self.to_latent = nn.Identity()

        # self.mlp_head = nn.Linear(dim, num_classes)
        self.detokenizers = nn.ModuleList([nn.Linear(dim, 1) for _ in range(12)])

    def forward(self, x):
        x_root = self.linear_embedding_root(x[:, 0:12])


        # 对每个node做embedding
        x_nodes = []
        for i in range(self.num_actions):
            node_input = torch.cat((x[:, 12+i].unsqueeze(1), x[:, 12+self.num_actions+i].unsqueeze(1), x[:, 12+2*self.num_actions+i].unsqueeze(1)), dim=1)
            x_node = self.linear_embedding_selfobs(node_input)
            x_nodes.append(x_node)
            # print("x_nodes",i,":", node_input)
        x_nodes = torch.stack(x_nodes, dim=1)
        
     
        x = torch.cat((x_nodes, x_root.unsqueeze(1)), dim=1)

        # print("after embedding:", x)
        b, n, _ = x.shape
        # print("x.shape:", x.shape)

        x += self.pos_embedding[:, :n]
        # position_ids = torch.arange(n, device=x.device).unsqueeze(0).expand(b, n)
        # x += self.pos_embedding(position_ids)

        x = self.transformer(x)
        
        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # x = self.to_latent(x)

        action = torch.zeros(x.shape[0], 12).to(x.device)
        for i in range(12):
            curr_action = self.detokenizers[i](x[:, i, :])
            action[:, i] = curr_action.squeeze(-1)

        return action


class Trf_Policy_Critic(nn.Module):
    def __init__(self, *,  input_dim, num_classes, dim, depth, heads, mlp_dim, activation, pool = 'cls', dim_head = 64):
        super().__init__()
        self.input_dim = input_dim  
        self.d_model = dim
        self.num_actions = num_classes
        self.activation = get_activation(activation)

        # # 定义不同类型输入的linear embedding
        # self.linear_embedding_actions = nn.Sequential(
        #     nn.Linear(self.num_actions, dim),
        #     self.activation,
        #     nn.LayerNorm(dim)
        # )
        # self.linear_embedding_root = nn.Sequential(
        #     nn.Linear(9, dim),
        #     self.activation,
        #     nn.LayerNorm(dim)
        # )
        # self.linear_embedding_commands = nn.Sequential(
        #     nn.Linear(3, dim),
        #     self.activation,
        #     nn.LayerNorm(dim)
        # )
        # self.linear_embedding_selfobs = nn.Sequential(
        #     nn.Linear(3, dim),  # 对于每个节点
        #     self.activation,
        #     nn.LayerNorm(dim)
        # )

        # 定义不同类型输入的linear embedding
        self.linear_embedding_root = nn.Linear(12, dim)
        self.linear_embedding_selfobs = nn.Linear(3, dim)
    
    
        # 位置编码
        max_seq_len = 1 + self.num_actions  # 3: actions, root, command; + self.num_actions for nodes


        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, dim))
        # self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim=dim)  # max_num_limbs

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        # encoder_layer =  nn.TransformerEncoderLayer(d_model=self.d_model, nhead=heads, dim_feedforward=mlp_dim, batch_first=True, dropout=0.)
        # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # self.pool = pool
        # self.to_latent = nn.Identity()

        # self.mlp_head = nn.Linear(dim, num_classes)

        self.detokenizers = nn.ModuleList([nn.Linear(dim, 1) for _ in range(12)])

    def forward(self, x):
        x_root = self.linear_embedding_root(x[:, 0:12])

        # 对每个node做embedding
        x_nodes = []
        for i in range(self.num_actions):
            node_input = torch.cat((x[:, 12+i].unsqueeze(1), x[:, 12+self.num_actions+i].unsqueeze(1), x[:, 12+2*self.num_actions+i].unsqueeze(1)), dim=1)
            x_node = self.linear_embedding_selfobs(node_input)
            x_nodes.append(x_node)
            # print("x_nodes",i,":", node_input)
        x_nodes = torch.stack(x_nodes, dim=1)
        
     
        x = torch.cat((x_nodes, x_root.unsqueeze(1)), dim=1)

        # print("after embedding:", x)
        b, n, _ = x.shape
        # print("x.shape:", x.shape)

        x += self.pos_embedding[:, :n]
        # position_ids = torch.arange(n, device=x.device).unsqueeze(0).expand(b, n)
        # x += self.pos_embedding(position_ids)

        x = self.transformer(x)
        
        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # x = self.to_latent(x)

        # x = self.mlp_head(x)

        value = torch.zeros(x.shape[0], 12).to(x.device)
        for i in range(12):
            curr_value = self.detokenizers[i](x[:, i, :])
            value[:, i] = curr_value.squeeze(-1)

        value = value.mean(dim = 1, keepdim=True)
        
        return value




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