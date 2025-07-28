from legged_gym.envs.base_unitree.legged_robot_config_unitree import LeggedRobotCfg_Unitree, LeggedRobotCfgPPO_Unitree

class G1RoughCfg( LeggedRobotCfg_Unitree ):
    class init_state( LeggedRobotCfg_Unitree.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_pitch_joint' : -0.1, 
           'left_hip_roll_joint' : 0,               
            'left_hip_yaw_joint' : 0. ,     
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_pitch_joint' : -0.1, 
           'right_hip_roll_joint' : 0, 
            'right_hip_yaw_joint' : 0.,                                 
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,       
           'torso_joint' : 0.
        }
    
    class env(LeggedRobotCfg_Unitree.env):
        num_observations = 48
        num_actions = 12
      

    class control( LeggedRobotCfg_Unitree.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 150,
                     'hip_roll': 150,
                     'hip_pitch': 150,
                     'knee': 300,
                     'ankle': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg_Unitree.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/urdf/g1.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["torso"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        dof_limit_org = {
                0: {'effort': 88.0, 'velocity': 32.0, 'lower': -2.35, 'upper': 3.05},   # left_hip_pitch_joint
                1: {'effort': 88.0, 'velocity': 32.0, 'lower': -0.26, 'upper': 2.53},   # left_hip_roll_joint
                2: {'effort': 88.0, 'velocity': 32.0, 'lower': -2.75, 'upper': 2.75},   # left_hip_yaw_joint
                3: {'effort': 139.0, 'velocity': 20.0, 'lower': -0.33489, 'upper': 2.5449},   # left_knee_joint
                4: {'effort': 40.0, 'velocity': 53.0, 'lower': -0.68, 'upper': 0.73},   # left_ankle_pitch_joint
                5: {'effort': 40.0, 'velocity': 53.0, 'lower': -0.2618, 'upper': 0.2618},   # left_ankle_roll_joint
                6: {'effort': 88.0, 'velocity': 32.0, 'lower': -2.35, 'upper': 3.05},   # right_hip_pitch_joint
                7: {'effort': 88.0, 'velocity': 32.0, 'lower': -2.53, 'upper': 0.26},   # right_hip_roll_joint
                8: {'effort': 88.0, 'velocity': 32.0, 'lower': -2.75, 'upper': 2.75},   # right_hip_yaw_joint
                9: {'effort': 139.0, 'velocity': 20.0, 'lower': -0.33489, 'upper': 2.5449},   # right_knee_joint
                10: {'effort': 40.0, 'velocity': 53.0, 'lower': -0.68, 'upper': 0.73},   # right_ankle_pitch_joint
                11: {'effort': 40.0, 'velocity': 53.0, 'lower': -0.2618, 'upper': 0.2618},   # right_ankle_roll_joint
        }

        custom_dof_limit = {
                0: {'effort': 88.0, 'velocity': 32.0, 'lower': -0.91, 'upper': 0.71},   # left_hip_pitch_joint
                1: {'effort': 88.0, 'velocity': 32.0, 'lower': -0.26, 'upper': 2.53},   # left_hip_roll_joint
                2: {'effort': 88.0, 'velocity': 32.0, 'lower': -2.75, 'upper': 2.75},   # left_hip_yaw_joint
                3: {'effort': 139.0, 'velocity': 20.0, 'lower': -0.33489, 'upper': 2.5449},   # left_knee_joint
                4: {'effort': 40.0, 'velocity': 53.0, 'lower': -0.68, 'upper': 0.73},   # left_ankle_pitch_joint
                5: {'effort': 40.0, 'velocity': 53.0, 'lower': -0.2618, 'upper': 0.2618},   # left_ankle_roll_joint
                6: {'effort': 88.0, 'velocity': 32.0, 'lower': -2.35, 'upper': 3.05},   # right_hip_pitch_joint
                7: {'effort': 88.0, 'velocity': 32.0, 'lower': -2.53, 'upper': 0.26},   # right_hip_roll_joint
                8: {'effort': 88.0, 'velocity': 32.0, 'lower': -2.75, 'upper': 2.75},   # right_hip_yaw_joint
                9: {'effort': 139.0, 'velocity': 20.0, 'lower': -0.33489, 'upper': 2.5449},   # right_knee_joint
                10: {'effort': 40.0, 'velocity': 53.0, 'lower': -0.68, 'upper': 0.73},   # right_ankle_pitch_joint
                11: {'effort': 40.0, 'velocity': 53.0, 'lower': -0.2618, 'upper': 0.2618},   # right_ankle_roll_joint
        }

  
    class rewards( LeggedRobotCfg_Unitree.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.728
        class scales( LeggedRobotCfg_Unitree.rewards.scales ):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-8
            feet_air_time = 1.0
            collision = 0.0
            action_rate = -0.01
            # torques = -0.0001
            dof_pos_limits = -5.0

class G1RoughCfgPPO( LeggedRobotCfgPPO_Unitree ):
    class policy:
        init_noise_std = 0.8
        nbodies = 14
        env_name = "g1"
        mu_activation = "tanh"
        actor_type = "transformer"
        critic_type = "transformer"
        embedding_dim = 64
        nheads = 2
        nlayers = 8
        dim_feedforward = 256
        is_mixed = False
        num_actor_obs = 51
        num_critic_obs = 48
        mask_pos = None
    class algorithm( LeggedRobotCfgPPO_Unitree.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO_Unitree.runner ):
        run_name = ''
        experiment_name = 'g1'

  
