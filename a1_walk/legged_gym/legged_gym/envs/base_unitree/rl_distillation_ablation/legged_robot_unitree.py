from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
from collections import OrderedDict, defaultdict
import itertools
import numpy as np
import os
import random
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import math
import torch
from torch import Tensor
from legged_gym.utils.terrain.terrain_unitree import Terrain_Unitree
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base_unitree.base_task_unitree import BaseTask_Unitree
from legged_gym.utils.math import wrap_to_pi
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config_unitree import LeggedRobotCfg_Unitree

class LeggedRobot_Unitree(BaseTask_Unitree):
    def __init__(self, cfg: LeggedRobotCfg_Unitree, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        self.saved_dof_limits = {}

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions, tmp_curr_iter):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self.curr_iter = tmp_curr_iter
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time-elapsed_time>0:
                    time.sleep(sim_time-elapsed_time)
            
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        
        # update curriculum
        if self.cfg.terrain.curriculum and (self.cfg.terrain.mesh_type in ['heightfield', 'trimesh']):
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_x_vel[env_ids] = 0.
        self.last_y_vel[env_ids] = 0.
        self.last_yaw_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(
                self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        
        
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def _get_proprioception_obs(self, privileged= False):
        return self.obs_super_impl[:, :self.cfg.env.num_observations]
    
    def _get_height_measurements_obs(self, privileged= False):
        return self.obs_super_impl[:, self.cfg.env.num_observations:(self.cfg.env.num_observations+187)]

    def _get_robot_config_obs(self, privileged= False):
        return self.robot_config_buffer

    def _get_is_dof_limit_obs(self, privileged= False):
        # return torch.zeros((self.num_envs, 3), device= self.sim_device)
        return torch.full((self.num_envs, 3), -1, device=self.sim_device)

    def _get_mask_obs_obs(self, privileged= False):
        return torch.zeros((self.num_envs, self.num_actions), device= self.sim_device)

    def _get_obs_from_components(self, components: list, privileged= False):
        obs_segments = self.get_obs_segment_from_components(components)
        obs = []
        for k, v in obs_segments.items():
            if k == "proprioception":
                obs.append(self._get_proprioception_obs(privileged))
            elif k == "height_measurements":
                obs.append(self._get_height_measurements_obs(privileged))
            else:
                # get the observation from specific component name
                # such as "_get_forward_depth_obs"
                obs.append(getattr(self, "_get_" + k + "_obs")(privileged))
        obs = torch.cat(obs, dim= 1)
        # print("obsshape:",obs.shape)
        # obs = torch.cat([obs, torch.zeros((obs.shape[0], 12), device=obs.device)], dim=1)
        return obs

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = cfg.noise.add_noise
        
        segment_start_idx = 0
        obs_segments = self.get_obs_segment_from_components(cfg.env.obs_components)
        # write noise for each corresponding component.
        for k, v in obs_segments.items():
            segment_length = np.prod(v)
            # write sensor scale to provided noise_vec
            # for example "_write_forward_depth_noise"
            # print(segment_start_idx,  segment_length)
            getattr(self, "_write_" + k + "_noise")(noise_vec[segment_start_idx: segment_start_idx + segment_length])
            segment_start_idx += segment_length

        return noise_vec
    
    def _write_proprioception_noise(self, noise_vec):
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands

    def _write_height_measurements_noise(self, noise_vec):
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:] = noise_scales.height_measurements * \
                noise_level * self.obs_scales.height_measurements

    def _write_robot_config_noise(self, noise_vec):
        if not hasattr(self.cfg.noise.noise_scales, "robot_config"):
            return
        noise_vec[:] = self.cfg.noise.noise_scales.robot_config * self.cfg.noise.noise_level * self.obs_scales.robot_config

    def _write_is_dof_limit_noise(self, noise_vec):
        if not hasattr(self.cfg.noise.noise_scales, "is_dof_limit"):
            return
        noise_vec[:] = 0.

    def _write_mask_obs_noise(self, noise_vec):
        if not hasattr(self.cfg.noise.noise_scales, "mask_obs"):
            return
        noise_vec[:] = 0.

    ##### defines observation segments, which tells the order of the entire flattened obs #####
    def get_obs_segment_from_components(self, components):
        """ Observation segment is defined as a list of lists/ints defining the tensor shape with
        corresponding order.
        """
        segments = OrderedDict()
        if "proprioception" in components:
            # print(self.cfg.env.num_observations)
            segments["proprioception"] = (self.cfg.env.num_observations,)
        if "height_measurements" in components:
            segments["height_measurements"] = (187,)
        if "robot_config" in components:
            """ Related to robot_config_buffer attribute, Be careful to change. """
            # robot shape friction
            # CoM (Center of Mass) x, y, z
            # base mass (payload)
            # motor strength for each joint
            segments["robot_config"] = (1 + 3 + 1 + 12,)
        # 添加 is_dof_limit 的处理
        if "is_dof_limit" in components:
            segments["is_dof_limit"] = (3,)  # 添加 3 位追踪 mask
        # 添加 mask_obs 的处理
        if "mask_obs" in components:
            segments["mask_obs"] = (self.num_actions,)  # 添加 12 位追踪 mask


        return segments

    def get_num_obs_from_components(self, components):
        obs_segments = self.get_obs_segment_from_components(components)
        num_obs = 0
        for k, v in obs_segments.items():
            num_obs += np.prod(v)
        return num_obs
    
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        # print("before get:", self.obs_buf.shape,self.obs_buf)
        self.obs_super_impl = self.obs_buf

        # print(self.obs_super_impl,self.obs_super_impl.shape)

        # 按照设置的模块增加obs_buf和privileged_obs_buf
        # actor obs
        self.obs_buf = self._get_obs_from_components(
            self.cfg.env.obs_components,
            privileged= False,
        )
        # print("after get:", self.obs_buf.shape,self.obs_buf)
        # print(self.obs_buf.shape)

        # critic obs
        if not self.num_privileged_obs is None:
            # print("1")
            self.privileged_obs_buf[:] = self._get_obs_from_components(
                self.cfg.env.privileged_obs_components,
                privileged= getattr(self.cfg.env, "privileged_obs_gets_privilege", False),
            )
            # print(self.privileged_obs_buf.shape)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        
        # print(self.cfg.asset.custom_dof_limit)
        # print(self.cfg.asset.dof_limit_org)
        # print(self.num_actions)
        envs_per_dof = max(1, self.num_envs // 4)
        # 定义关节的映射关系，包含 indices 和 dof_indices

        if self.cfg.asset.name == "h1":
            joint_mapping_for_obs = {
            'left_hip_yaw_joint': [[12, 22, 32], [0]],
            'left_hip_roll_joint': [[13, 23, 33], [1]],
            'left_hip_pitch_joint': [[14, 24, 34], [2]],
            'left_knee_joint': [[15, 25, 35], [3]],
            'left_ankle_joint': [[16, 26, 36], [4]],
            'right_hip_yaw_joint': [[17, 27, 37], [5]],
            'right_hip_roll_joint': [[18, 28, 38], [6]],
            'right_hip_pitch_joint': [[19, 29, 39], [7]],
            'right_knee_joint': [[20, 30, 40], [8]],
            'right_ankle_joint': [[21, 31, 41], [9]],
            }
        elif self.cfg.asset.name == "g1":
            # print("1")
            joint_mapping_for_obs = {
            'left_hip_pitch_joint': ([12, 24, 36], [0]),
            'left_hip_roll_joint': ([13, 25, 37], [1]),
            'left_hip_yaw_joint': ([14, 26, 38], [2]),
            'left_knee_joint': ([15, 27, 39], [3]),
            'left_ankle_pitch_joint': ([16, 28, 40], [4]),
            'left_ankle_roll_joint': ([17, 29, 41], [5]),
            'right_hip_pitch_joint': ([18, 30, 42], [6]),
            'right_hip_roll_joint': ([19, 31, 43], [7]),
            'right_hip_yaw_joint': ([20, 32, 44], [8]),
            'right_knee_joint': ([21, 33, 45], [9]),
            'right_ankle_pitch_joint': ([22, 34, 46], [10]),
            'right_ankle_roll_joint': ([23, 35, 47], [11]),
            }
        else:
            raise ValueError("Unexpected asset name!")


        # 遍历每个环境并应用 mask
        custom_dof_limit = {dof_index: {
            'effort': torch.tensor(limits['effort']),
            'velocity': torch.tensor(limits['velocity']),
            'lower': torch.tensor(limits['lower']),
            'upper': torch.tensor(limits['upper'])
        } for dof_index, limits in self.cfg.asset.custom_dof_limit.items()}

        # print(custom_dof_limit)
        # 计算每个 DOF 的 upper 和 lower 差值并存储在数组中
        dof_differences = []

        # 遍历12个DOF并计算差值
        for dof_index in range(self.num_actions):
            lower = custom_dof_limit[dof_index]['lower']
            upper = custom_dof_limit[dof_index]['upper']
            difference = (upper - lower) / 2  # 差值的一半
            dof_differences.append(difference)

        for i in range(self.num_envs):  
            dof_index = 3
            # Determine which dof group this environment belongs to
            if i < (3 * envs_per_dof):
                dof_index = i // envs_per_dof

            # GROUP1动态且随机选择不超过3个DOF损坏
            if self.cfg.obs.dynamic_fault_obs and dof_index==0:
                if self.curr_iter >= 0:  # 当迭代次数达到阈值后，开始加入损坏
                    if i in self.dof_limit_dict:  # 如果当前环境ID存在于字典中
                        faulty_dof_indices = self.dof_limit_dict[i]
                        # print(i,dof_index,faulty_dof_indices)
                        for tmp_dof_index in faulty_dof_indices:             
                            # 遍历 joint_mapping_for_obs 找到对应的 DOF
                            for dof_name, (indices, dof_indices) in joint_mapping_for_obs.items():
                                if tmp_dof_index in dof_indices:
                                    # 模拟损坏，obs 置0，设置检测标记
                                    self.obs_buf[i, (-self.num_actions + tmp_dof_index)] = 1
                                    for idx in indices:
                                        self.obs_buf[i, idx] = 0  # 将观测置为0，表示损坏
                                    break
                    else :
                         raise ValueError(f"Current ENV: {i} has no faults, which is not normal. ")

            # GROUP2动态且随机选择不超过3个DOF损坏
            if self.cfg.obs.dynamic_fault_obs and dof_index==1:
                if self.curr_iter >= 0:  # 当迭代次数达到阈值后，开始加入损坏
                    # 随机选择不超过3个 DOF
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

            # GROUP3 保持原样

            # GROUP4 动态且随机选择不超过3个DOF损坏+无法检测
            

        # print(self.obs_buf[:, -(self.num_actions+3):-self.num_actions])
        # print(self.obs_buf[:,-13:])
        # print(self.obs_buf.shape)
        # print("obsbuf:",self.obs_buf[:, -13:])

    def create_sim(self):
        """ Creates simulation, terrain and environments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain_Unitree(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)



    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

            # Add robot_config
            if env_id == 0:
                all_obs_components = self.all_obs_components
                if "robot_config" in all_obs_components:
                    all_obs_components
                    self.robot_config_buffer = torch.empty(
                        self.num_envs, 1 + 3 + 1 + 12,
                        dtype= torch.float32,
                        device= self.device,
                    )
        
            if hasattr(self, "robot_config_buffer"):
                self.robot_config_buffer[env_id, 0] = props[0].friction
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props


    # def _reprocess_dof_props(self, props, env_id):
    #     """ Callback allowing to store/change/randomize the DOF properties of each environment.
    #         Called During environment creation.
    #         Base behavior: stores position, velocity and torques limits defined in the URDF

    #     Args:
    #         props (numpy.array): Properties of each DOF of the asset
    #         env_id (int): Environment id

    #     Returns:
    #         [numpy.array]: Modified DOF properties
    #     """
    #     for i in range(len(props)):
    #         self.dof_pos_limits[env_id, i, 0] = props["lower"][i].item()
    #         self.dof_pos_limits[env_id, i, 1] = props["upper"][i].item()
    #         self.dof_vel_limits[env_id, i] = props["velocity"][i].item()
    #         self.torque_limits[env_id, i] = props["effort"][i].item()
    #     return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])

        if hasattr(self, "robot_config_buffer"):
            self.robot_config_buffer[env_id, 1] = props[0].com.x
            self.robot_config_buffer[env_id, 2] = props[0].com.y
            self.robot_config_buffer[env_id, 3] = props[0].com.z
            self.robot_config_buffer[env_id, 4] = props[0].mass
            self.robot_config_buffer[env_id, 5:5+12] = self.motor_strength[env_id] if hasattr(self, "motor_strength") else 1.
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

   
    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(
            self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(
            self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(
            self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(
            get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch(
            [1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions,
                                   dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(
            self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(
            self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions,
                                   dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                    device=self.device, requires_grad=False)  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel,
                                           self.obs_scales.ang_vel], device=self.device, requires_grad=False,)  # TODO change this
        self.feet_air_time = torch.zeros(
            self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(
            self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(
            self.base_quat, self.gravity_vec)
        self.x_vel = self.base_lin_vel[:, 0]
        self.last_x_vel = torch.zeros_like(self.x_vel)
        self.y_vel = self.base_lin_vel[:, 1]
        self.last_y_vel = torch.zeros_like(self.y_vel)
        self.yaw_vel = self.base_lin_vel[:, 2]
        self.last_yaw_vel = torch.zeros_like(self.yaw_vel)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(
                        f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
      

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        print(self.num_dofs,self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        # 暂存当前随机数生成器的状态
        saved_state = random.getstate()
        random.seed(42)
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            # print(self.dof_pos_limits)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

            # 生成2-4个随机dof损坏
            # 随机选择若干个DOF
            if self.cfg.asset.name == 'h1':
                num_faulty_dofs = random.randint(2, 3)  # 生成3或4的随机整数
            elif self.cfg.asset.name == 'g1':
                num_faulty_dofs = random.randint(2, 4)  # 生成3或4的随机整数
            # num_faulty_dofs = 2
            # 从0到11中随机选取
            faulty_dof_indices = random.sample(range(self.num_actions), num_faulty_dofs)  # 从0到11中不重复地选择num_faulty_dofs个元素
            # print(faulty_dof_indices)
            self.dof_limit_state.append((i, faulty_dof_indices))
        # print(self.dof_pos_limits)
         # 在 create_env 函数执行完后，将 dof_limit_state 转为字典
        self.dof_limit_dict = {env_id: faulty_dof_indices for (env_id, faulty_dof_indices) in self.dof_limit_state}
        random.setstate(saved_state)

        if self.cfg.asset.set_dof_limit_flag:

            envs_per_dof = max(1, self.num_envs // 4)


            for i in range(self.num_envs): 

                # 4 group
                dof_index = 3
                # Determine which dof group this environment belongs to
                if i < (3 * envs_per_dof):
                    dof_index = i // envs_per_dof
                print(i, dof_index)

                if envs_per_dof!=0 and dof_index==1:
                    tmp_dof_props = self.gym.get_actor_dof_properties(self.envs[i], self.actor_handles[i])
                    # print(tmp_dof_props)

                    if i in self.dof_limit_dict:  # 如果当前环境ID存在于字典中
                        faulty_dof_indices = self.dof_limit_dict[i]
                        print(faulty_dof_indices)
                        for faulty_dof_index in faulty_dof_indices:     
                            # custom dof limits
                            tmp_dof_props['effort'][faulty_dof_index] = self.cfg.asset.custom_dof_limit[faulty_dof_index]['effort']
                            tmp_dof_props['velocity'][faulty_dof_index] = self.cfg.asset.custom_dof_limit[faulty_dof_index]['velocity']
                            tmp_dof_props['lower'][faulty_dof_index] = self.cfg.asset.custom_dof_limit[faulty_dof_index]['lower']
                            tmp_dof_props['upper'][faulty_dof_index] = self.cfg.asset.custom_dof_limit[faulty_dof_index]['upper']
                    else:
                         raise ValueError(f"Current ENV: {i} has no faults, which is not normal. ")
                    
                    # 应用更新的 DOF 属性
                    # print(tmp_dof_props['effort'][faulty_dof_index])
                    # print(tmp_dof_props['velocity'][faulty_dof_index])
                    # modified_dof_props = self._process_dof_props(tmp_dof_props, i)
                    self.gym.set_actor_dof_properties(self.envs[i], self.actor_handles[i], tmp_dof_props)

                    # tmp = self.gym.get_actor_dof_properties(self.envs[i], self.actor_handles[i])
                    # print(i, tmp)
                elif envs_per_dof!=0 and dof_index==3:
                    tmp_dof_props = self.gym.get_actor_dof_properties(self.envs[i], self.actor_handles[i])
                    if i in self.dof_limit_dict:  # 如果当前环境ID存在于字典中
                        faulty_dof_indices = self.dof_limit_dict[i]
                        print(faulty_dof_indices)
                        for faulty_dof_index in faulty_dof_indices:     
                            # custom dof limits
                            tmp_dof_props['effort'][faulty_dof_index] = self.cfg.asset.custom_dof_limit[faulty_dof_index]['effort']
                            tmp_dof_props['velocity'][faulty_dof_index] = self.cfg.asset.custom_dof_limit[faulty_dof_index]['velocity']
                            # tmp_dof_props['lower'][faulty_dof_index] = self.cfg.asset.custom_dof_limit[faulty_dof_index]['lower']
                            # tmp_dof_props['upper'][faulty_dof_index] = self.cfg.asset.custom_dof_limit[faulty_dof_index]['upper']
                    else:
                         raise ValueError(f"Current ENV: {i} has no faults, which is not normal. ")
                    
                    # 应用更新的 DOF 属性
                    # print(tmp_dof_props['effort'][faulty_dof_index])
                    # print(tmp_dof_props['velocity'][faulty_dof_index])
                    #  modified_dof_props = self._process_dof_props(tmp_dof_props, i)
                    self.gym.set_actor_dof_properties(self.envs[i], self.actor_handles[i], tmp_dof_props)
                elif envs_per_dof!=0 and (dof_index==0 or dof_index==2):
                    if i in self.dof_limit_dict:  # 如果当前环境ID存在于字典中
                        faulty_dof_indices = self.dof_limit_dict[i]
                        print(faulty_dof_indices)
                        print(f"NO DOF_LIMITS ARE ADDED SINCE DOF NEEDS NO LIMITS for environment: {i}")
                    else:
                         raise ValueError(f"Current ENV: {i} has no faults, which is not normal. ")
                else:
                    print(f"NO DOF_LIMITS ARE ADDED SINCE ENV_NUM IS BELOW DOF_NUM for environment: {i}")

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(
                0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (
                self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(
                self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels,
                                                       self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(
                num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
     

        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)


    #------------ reward functions----------------
#     def _reward_lin_vel_z(self):
#         # Penalize z axis base linear velocity
#         return torch.square(self.base_lin_vel[:, 2])
    
#     def _reward_ang_vel_xy(self):
#         # Penalize xy axes base angular velocity
#         return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
#     def _reward_orientation(self):
#         # Penalize non flat base orientation
#         return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

#     def _reward_base_height(self):
#         # Penalize base height away from target
#         base_height = torch.mean(self.root_states[:, 2].unsqueeze(
#             1) - self.measured_heights, dim=1)
#         return torch.square(base_height - self.cfg.rewards.base_height_target)
    
#     def _reward_torques(self):
#         # Penalize torques
#         return torch.sum(torch.square(self.torques), dim=1)

#     def _reward_dof_vel(self):
#         # Penalize dof velocities
#         return torch.sum(torch.square(self.dof_vel), dim=1)
    
#     def _reward_dof_acc(self):
#         # Penalize dof accelerations
#         return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
#     def _reward_action_rate(self):
#         # Penalize changes in actions
#         return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
#     def _reward_collision(self):
#         # Penalize collisions on selected bodies
#         return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
#     def _reward_termination(self):
#         # Terminal reward / penalty
#         return self.reset_buf * ~self.time_out_buf
    
#     def _reward_dof_pos_limits(self):
#         # Penalize dof positions too close to the limit
#         out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
#         out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
#         return torch.sum(out_of_limits, dim=1)

#     def _reward_dof_vel_limits(self):
#         # Penalize dof velocities too close to the limit
#         # clip to max error = 1 rad/s per joint to avoid huge penalties
#         return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

#     def _reward_torque_limits(self):
#         # penalize torques too close to the limit
#         return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

#     def _reward_tracking_lin_vel(self):
#         # Tracking of linear velocity commands (xy axes)
#         lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
#         return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
#     def _reward_tracking_ang_vel(self):
#         # Tracking of angular velocity commands (yaw) 
#         ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
#         return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

#     def _reward_feet_air_time(self):
#         # Reward long steps
#         # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
#         contact = self.contact_forces[:, self.feet_indices, 2] > 1.
#         contact_filt = torch.logical_or(contact, self.last_contacts) 
#         self.last_contacts = contact
#         first_contact = (self.feet_air_time > 0.) * contact_filt
#         self.feet_air_time += self.dt
#         rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
#         rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
#         self.feet_air_time *= ~contact_filt
#         return rew_airTime
    
#     def _reward_stumble(self):
#         # Penalize feet hitting vertical surfaces
#         return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
#              5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
#     def _reward_stand_still(self):
#         # Penalize motion at zero commands
#         return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

#     def _reward_feet_contact_forces(self):
#         # penalize high contact forces
#         return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.root_states[:, 2]
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    # def _reward_base_height(self):
    #     # Penalize base height away from target
    #     base_height = torch.mean(self.root_states[:, 2].unsqueeze(
    #         1) - self.measured_heights, dim=1)
    #     return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        # print(self.dof_pos.shape, self.dof_pos)
        # print(self.dof_pos_limits.shape, self.dof_pos_limits)
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)



##### Some helper functions that override parent class attributes #####
    @property
    def all_obs_components(self):
        components = set(self.cfg.env.obs_components)
        if getattr(self.cfg.env, "privileged_obs_components", None):
            components.update(self.cfg.env.privileged_obs_components)
        return components
    
    @property
    def obs_segments(self):
        return self.get_obs_segment_from_components(self.cfg.env.obs_components)
    @property
    def privileged_obs_segments(self):
        components = getattr(
            self.cfg.env,
            "privileged_obs_components",
            None
        )
        if components is None:
            return None
        else:
            return self.get_obs_segment_from_components(components)
    @property
    def num_obs(self):
        """ get this value from self.cfg.env """
        assert "proprioception" in self.cfg.env.obs_components, "missing critical observation component 'proprioception'"
        return self.get_num_obs_from_components(self.cfg.env.obs_components)
    @num_obs.setter
    def num_obs(self, value):
        """ avoid setting self.num_obs """
        pass
    @property
    def num_privileged_obs(self):
        """ get this value from self.cfg.env """
        components = getattr(
            self.cfg.env,
            "privileged_obs_components",
            None
        )
        if components is None:
            return None
        else:
            return self.get_num_obs_from_components(components)
    @num_privileged_obs.setter
    def num_privileged_obs(self, value):
        """ avoid setting self.num_privileged_obs """
        pass