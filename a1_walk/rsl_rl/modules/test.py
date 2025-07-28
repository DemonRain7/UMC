import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic import ActorCritic

# 加载模型的.pt文件
# model = torch.load('your_model.pt')


model = ActorCritic(num_actions = 12, env_name = 'a1', nbodies = 14, num_actor_obs = 51, num_critic_obs = 48, actor_type='transformer', critic_type='transformer', embedding_dim=120,
                        nheads=2,
                        nlayers=4,
                        dim_feedforward=128,mu_activation = "tanh")  # 根据你的模型类实例化
model = torch.load('/home/Rain/Rain_tmp/BodyTransformer/a1_walk/rsl_rl/modules/model_5000.pt')

# print(model)

# print(model.keys())  

print(model['model_state_dict'].keys())

# 假设模型中包含一个 MultiheadAttention 层
for layer in model['model_state_dict'].actor.net.encoder.layers:  # 迭代所有层
    if isinstance(layer.attn, nn.MultiheadAttention):
        # 确保每个 MultiheadAttention 层的 need_weights 被设置为 True
        layer.attn.need_weights = True

torch.save(model, '/home/Rain/Rain_tmp/BodyTransformer/a1_walk/rsl_rl/modules/modified_model.pt')