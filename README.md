# SafeRL-Trf
 
### Installation ###
1. Create a new python virtual env with python 3.8 
2. Install pytorch 2.1.2 with cuda-12.1:
    - `pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121`
3. Install Isaac Gym
   - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
   - `cd isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs `isaacgym/docs/index.html`)
4. Install rsl_rl (PPO implementation)
   -  `cd rsl_rl && pip install -e .`
   -  * some problems exists, if this does not work, then download rsl_rl from github and git checkout v1.0.2
5. Install legged_gym
    - Clone this repository
   - `cd legged_gym && pip install -e .`
6. Install Tensorboard
   - pip install tensorboard
7. Install Positional Encodings
   - pip install positional_encodings

### Before Running ###
1. Set the code environment:
   export PYTHONPATH=/home/Rain/Rain_tmp/BodyTransformer/a1_walk/rsl_rl:$PYTHONPATH
   export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH (If using Conda)

2. If you are using numpy>=1.20, just replace np.float in the isaacgym/python/isaacgym/torch_utils.py into float.


### TIPS ###
1.A1 WALK的代码在legged_gym/envs/base以及env/a1下；H1任务的代码在legged_gym/envs/base_unitree和env/h1下。我们项目代码修改的部分主要看compute_observation和_create_env类即可，为了方便翻阅，我加了个NOTE，直接Crtl+F搜索NOTE可以快速找到这两个Class！

2.rsl_rl/modules/actor_critic.py中可以看到Transformer的应用，具体实现请见rsl_rl/modules/transformer.py，其他的transformer_gx6啊之类的都是细节不同的transformer，要参考transformer架构的话直接看transformer即可；如果想了解H1机器人我是如何共享tokenize的线性层的就去看transformer_gx6的ActorObsTokenizer和CriticObsTokenizer类；如果想了解我是如何共享detokenizer的线性层的就去看transformer_gxdetokenizer的ActionDetokenizer和ValueDetokenizer类即可。具体transformer架构我会在后续readme中介绍




### 如何实现的分组obs置0 ###
这部分就是解释compute_observation的代码是如何把检测到的obs置为0的，不需要看跳过即可！
完整的obs_buf由以下几大类组成：
 obs_components = [
            "proprioception", # 12+3*self.num_actions，这个是原本就有的：base_lin_vel、base_ang_vel、projected_gravity、commands[:, :3]、dof_pos、dof_vel和actions
            "height_measurements", # 187，这个就是heights，也是原本的没有做过更改
            "is_dof_limit", # 3，这个是我加的，全-1就表示没有损坏，全1就表示存在至少一个dof是损坏的
            "mask_obs", # self.num_actions，这个并不会直接输入到transformer encoder中，仅仅是用来表示是哪一位dof出了问题，方便进行mask处理用的
        ]

详细步骤：
1.先计算出每一组多少个env，随后在Step2中由dof_index计算得出env对应分组。共4组：（此部分代码在legged_gym/envs/base/legged_robot_field.py（a1任务）/legged_gym/envs/base_unitree/legged_robot_unitree.py（h1任务）中）

```python
  envs_per_dof = max(1, self.num_envs // 4)  # 每组的环境数量
```

2.然后准备map，在我们想对某一个dof的所有obs都置为0（代表obs损坏）的时候会调用到，其中对应分组便是对应了某一dof的obs表示，置0即为mask。（此部分代码在legged_gym/envs/base/legged_robot_field.py中）
举例说明：对于env=1，如果我们想把第0给dof给损坏了，那么在加上了limit之后我们就把obs_buf中对应的self_obsbuf[1,12],self_obsbuf[1,24],self_obsbuf[1,36]都置为0，同时obs的mask_obs部分（此处为后12位的第0位）对应置为1。
下列仅仅展示obs，priviledged_obs同理
```python
   # 定义关节的映射关系，包含 indices 和 dof_indices
   joint_mapping_for_obs = {
         'left_hip_yaw_joint': ([12, 22, 32], [0]),
         'left_hip_roll_joint': ([13, 23, 33], [1]),
         'left_hip_pitch_joint': ([14, 24, 34], [2]),
         'left_knee_joint': ([15, 25, 35], [3]),
         'left_ankle_joint': ([16, 26, 36], [4]),
         'right_hip_yaw_joint': ([17, 27, 37], [5]),
         'right_hip_roll_joint': ([18, 28, 38], [6]),
         'right_hip_pitch_joint': ([19, 29, 39], [7]),
         'right_knee_joint': ([20, 30, 40], [8]),
         'right_ankle_joint': ([21, 31, 41], [9]),
         }

```

3.然后就是把之前在_create_env中初始化随机的那几个dof给从self.dof_limit_dict读取出来，比如env=1的第0，1个dof坏了，那么self.dof_limit_dict[1]就能读取出0，1，然后我们就对0，1dof对应的"mask_obs"置1（方便第四步进行mask）、"proprioception"中dof对应的3个obs位置为0、"is_dof_limit"置为111（表示有dof损坏；如果是仅仅obs损坏的情况下呢，就这里不动，保持-1-1-1）

```python
  if self.cfg.obs.dynamic_fault_obs and dof_index==1:
           if self.curr_iter >= 0:  # 当迭代次数达到阈值后，开始加入损坏
                    # Determine which dof group this environment belongs to
                    
                    current_dof_props = self.gym.get_actor_dof_properties(self.envs[i], self.actor_handles[i])
                    current_dof_props_tensors = {
                        dof_index: {
                            'effort': torch.tensor(prop['effort']),
                            'velocity': torch.tensor(prop['velocity']),
                            'lower': torch.tensor(prop['lower']),
                            'upper': torch.tensor(prop['upper'])
                        }
                        for dof_index, prop in enumerate(current_dof_props)
                    }
                    # 使用 self.dof_limit_dict 中保存的损坏信息进行处理
                    if i in self.dof_limit_dict:  # 如果当前环境ID存在于字典中
                        faulty_dof_indices = self.dof_limit_dict[i]
                        # print(i,dof_index,faulty_dof_indices)
                        for tmp_dof_index in faulty_dof_indices:
                            # 检查当前DOF是否设置了限制
                            # print(current_dof_props_tensors[tmp_dof_index]['effort'])
                            if ((current_dof_props_tensors[tmp_dof_index]['effort'] == custom_dof_limit[tmp_dof_index]['effort']) and (current_dof_props_tensors[tmp_dof_index]['velocity'] == custom_dof_limit[tmp_dof_index]['velocity'])):
                                for dof_name, (indices, dof_indices) in joint_mapping_for_obs.items():
                                    if tmp_dof_index in dof_indices:
                                        self.obs_buf[i, (-self.num_actions + tmp_dof_index)] = 1  # 标记第 dof_index 位
                                        self.obs_buf[i, -(self.num_actions + 3):-self.num_actions] = 1  # 表示已经损坏
                                        # self.obs_buf[i, -3:] = 1  # 表示已经损坏
                        
                                        for idx in indices:
                                            self.obs_buf[i, idx] = 0  # 设置 obs 为0，表示损坏
                                        break

                            else:
                                raise ValueError(f"DOF properties at index {tmp_dof_index} do not match custom limits. "
                                    f"Current: {current_dof_props_tensors[tmp_dof_index]}, "
                                    f"Custom: {custom_dof_limit[tmp_dof_index]}")

                    else :
                         raise ValueError(f"Current ENV: {i} has no faults, which is not normal. ")
```


4.Actor和Critic模型中进行检测，这里就相当于读取之前说的"mask_obs"部分。如果第i个env的第m位坏掉，则设mask[i,m]=True后传入到Transformer Encoder中，代表i这个env的第m个dof被mask掉。（此部分代码在rsl_rl/modules/transformer.py中）
```python
   batch_size, nbodies = x.shape[0], 14  # TODO:假设 nbodies 是 12，需根据实际情况调整
        # Initialize mask
        mask = torch.zeros((batch_size, nbodies), dtype=torch.bool, device=x.device)

        # TODO:NOT HARD-CODED
        # Convert x to a boolean tensor where True represents values equal to 1 (masked)
        bool_x = (x[:, -12:] == 1)  # Check last 12 bits of x

        # Assign mask for each environment based on DOF mask bits
        mask[:, :12] = bool_x
```
```python
  if mask is not None:
            # Pass the masked x to the Transformer encoder
            x = self.encoder(x, src_key_padding_mask=mask)
        else:
            x = self.encoder(x)
```

### Transformer模型整体结构讲解 ###
整体架构为：Tokenizer-Encoder-Detokenizer。这里以rsl_rl/modules/transformer.py为例子写在下面，方便知晓整体结构：
```python
class BodyActor(nn.Module):
    def __init__(self, name, net, embedding_dim, action_dim, stack_time=True, global_input=False, mu_activation=None, device='cuda',nbodies = 14):
        super(BodyActor, self).__init__()
        self.global_input = global_input
        # self.tokenizer = ObsTokenizer(name, embedding_dim, stack_time, device)
        self.tokenizer = ActorObsTokenizer(dim = embedding_dim, num_actions=action_dim, nbodies = nbodies)
        self.net = net
        self.detokenizer = ActionDetokenizer(name, net.output_dim, action_dim, global_input, device=device)

        self.tokenizer.to(device)
        self.net.to(device)
        self.detokenizer.to(device)

        self.mu_activation = mu_activation

    def forward(self, x):
        # print("input actor:", x.shape)
        x, mask = self.tokenizer(x) # (B, nbodies, lookback_steps, embedding_dim)
        # 判断 mask 是否全为 False
        if torch.all(mask == False):  # 或者 torch.all(~mask)
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
```

Tokenzier部分：
```python
class ActorObsTokenizer(nn.Module):
    def __init__(self, *, dim, num_actions, nbodies=14):
        super().__init__()
        self.activation = nn.ELU()
        self.num_actions = num_actions
        self.nb = nbodies

       
        # 定义不同类型输入的linear embedding
        self.linear_embedding_selfobs = nn.ModuleList(
            [nn.Linear(3, dim) for _ in range(self.num_actions)]
        )
        # 定义不同类型输入的linear embedding
        self.linear_embedding_root = nn.Linear(12, dim)
        self.linear_embedding_heights = nn.Linear(187, dim)
        self.linear_embedding_isdoflimit = nn.Linear(3, dim)

    def forward(self, x):
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
        x_heights = self.linear_embedding_heights(x[:, (12+3*self.num_actions):(12+3*self.num_actions+187)])
        x_isdoflimit = self.linear_embedding_isdoflimit(x[:, -(self.num_actions+3):-self.num_actions])

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
            x_node = self.linear_embedding_selfobs[i](node_input)  # 使用不同的线性层
            x_nodes.append(x_node)
        x_nodes = torch.stack(x_nodes, dim=1)
        x = torch.cat((x_nodes, x_root.unsqueeze(1), x_heights.unsqueeze(1), x_isdoflimit.unsqueeze(1)), dim=1)
        
        return x, mask


class CriticObsTokenizer(nn.Module):
    def __init__(self, *, dim, num_actions,nbodies = 14):
        super().__init__()
        self.activation = nn.ELU()
        self.num_actions = num_actions
        self.nb = nbodies - 1 # 无法检测到机器人内部损坏

        
        self.linear_embedding_selfobs = nn.ModuleList(
            [nn.Linear(3, dim) for _ in range(self.num_actions)]
        )
        # 定义不同类型输入的linear embedding
        self.linear_embedding_root = nn.Linear(12, dim)
        self.linear_embedding_heights = nn.Linear(187, dim)

    def forward(self, x):
        # max_value = torch.max(x)
        # print("max:", max_value)
        batch_size, nbodies = x.shape[0], self.nb   # TODO:假设 nbodies 是 14，需根据实际情况调整
        # Process x with linear embeddings
        x_root = self.linear_embedding_root(x[:, 0:12])
        x_heights = self.linear_embedding_heights(x[:, (12+3*self.num_actions):(12+3*self.num_actions+187)])

        # 对每个node做embedding
        x_nodes = []
        for i in range(self.num_actions):
            node_input = torch.cat((
                x[:, 12 + i].unsqueeze(1),
                x[:, 12 + self.num_actions + i].unsqueeze(1),
                x[:, 12 + 2 * self.num_actions + i].unsqueeze(1)
            ), dim=1)
            x_node = self.linear_embedding_selfobs[i](node_input)  # 使用不同的线性层
            x_nodes.append(x_node)

        x_nodes = torch.stack(x_nodes, dim=1)
        x = torch.cat((x_nodes, x_heights.unsqueeze(1), x_root.unsqueeze(1)), dim=1)
        
        return x
```
Encoder部分：
```python
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
```


Detokenizer部分：
```python
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
```
