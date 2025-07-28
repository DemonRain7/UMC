# rld

Train:
nbodies = 15
有两个phase，注意设置num_steps_per_env = 50;
phase1的时候，actor type = "transformer_rma"；phase2的时候actor和critic type = "transformer_rma_phase2"
此外，transformer要改成trf_rld；base下改成rld的
7400Env

Test:
play改成play_rld；base下换成play;
如果要换pt模型路径的话，修改take_registry.py中的make_alg_runner_for_adapt_play
注意改config
4000Env


# curriculum
Train:
nbodies = 14
num_steps_per_env = 24
obs里的curriculum_training: true, init_dmg_flag: 1
base里换一下

Test:
transformer就用a1即可，play和base里的play都不变；

# TFQL
Train:
actor_type = "mlp_tfql", critic是"mlp_tfql"
obs_component加上"dmg_flag"
nbodies = 15
注意设置num_steps_per_env = 50;
此外，transformer要改成trf_tfql；base下改成tfql的
只能num_env = 4096

Test:
play记得成tfql的
transformer要改成trf_tfql
base下play，然后cfg里obs_slice换成"mask_obs"

改config：
# A1
地形6 6 4； 渲染180000000； 两个true； custom dof需要修改
perlin noise要改



# SMS
1、训练参数：
num_env_per_step = 24
        nbodies = 1
        env_name = "a1"
        mu_activation = "tanh"
        actor_type = "transformer" # TODO: IF transformer_rma_phase2/mlp_tfql, step=50, else step=24!!!!!
        critic_type = "mlp" # TODO: IF transformer_rma_phase2, step=50, else step=24!!!!!
        embedding_dim = 512
        nheads = 2
        nlayers = 16
        dim_feedforward = 256
        is_mixed = False
        num_actor_obs = 51
        num_critic_obs = 48
        mask_pos = None
2、actor用transformer_sms，critic用mlp/transformer_sms；

Play的时候：
1、首先用play_sms文件夹；


