from legged_gym.envs.base_unitree.legged_robot_config_unitree import LeggedRobotCfg_Unitree, LeggedRobotCfgPPO_Unitree

class H1RoughCfg( LeggedRobotCfg_Unitree ):
    class init_state( LeggedRobotCfg_Unitree.init_state ):
        pos = [0.0, 0.0, 1.0] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.4,         
           'left_knee_joint' : 0.8,       
           'left_ankle_joint' : -0.4,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.4,                                       
           'right_knee_joint' : 0.8,                                             
           'right_ankle_joint' : -0.4,                                     
           'torso_joint' : 0., 
           'left_shoulder_pitch_joint' : 0., 
           'left_shoulder_roll_joint' : 0, 
           'left_shoulder_yaw_joint' : 0.,
           'left_elbow_joint'  : 0.,
           'right_shoulder_pitch_joint' : 0.,
           'right_shoulder_roll_joint' : 0.0,
           'right_shoulder_yaw_joint' : 0.,
           'right_elbow_joint' : 0.,
        }
    
    class env(LeggedRobotCfg_Unitree.env):
        num_observations = 42
        num_actions = 10
      

    class control( LeggedRobotCfg_Unitree.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 200,
                     'hip_roll': 200,
                     'hip_pitch': 200,
                     'knee': 300,
                     'ankle': 40,
                     'torso': 300,
                     'shoulder': 100,
                     "elbow":100,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 5,
                     'hip_roll': 5,
                     'hip_pitch': 5,
                     'knee': 6,
                     'ankle': 2,
                     'torso': 6,
                     'shoulder': 2,
                     "elbow":2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg_Unitree.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/urdf/h1.urdf'
        name = "h1"
        foot_name = "ankle"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg_Unitree.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.98
        class scales( LeggedRobotCfg_Unitree.rewards.scales ):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -1.0
            orientation = -1.0
            base_height = -100.0
            dof_acc = -3.5e-8
            feet_air_time = 1.0
            collision = 0.0
            action_rate = -0.01
            torques = 0.0
            dof_pos_limits = -10.0

class H1RoughCfgPPO( LeggedRobotCfgPPO_Unitree ):
    class algorithm( LeggedRobotCfgPPO_Unitree.algorithm ):
        entropy_coef = 0.01
    class policy( LeggedRobotCfgPPO_Unitree.policy ):
        nbodies = 12
        env_name = "h1"
        mu_activation = "tanh"
        actor_type = "transformer"
        critic_type = "transformer"
        embedding_dim = 120
        nheads = 2
        nlayers = 4
        dim_feedforward = 128
        is_mixed = False
        num_actor_obs = 45
        num_critic_obs = 42
        mask_pos = None
    class runner( LeggedRobotCfgPPO_Unitree.runner ):
        run_name = ''
        experiment_name = 'h1'

  
