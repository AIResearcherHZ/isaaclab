# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Taks_T1 AMP环境实现 - 与S2S2R完全兼容"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from .taks_t1_amp_env_cfg import TaksT1AmpEnvCfg
from .motions import MotionLoader

# 训练顺序的32个关节名称（与S2S2R/IO描述一致）
TRAINING_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "neck_yaw_joint", "right_shoulder_pitch_joint", "left_ankle_pitch_joint",
    "right_ankle_pitch_joint", "left_shoulder_roll_joint", "neck_roll_joint",
    "right_shoulder_roll_joint", "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "neck_pitch_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint",
    "right_wrist_roll_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
]

# 默认关节位置（训练顺序）
DEFAULT_DOF_POS = torch.tensor([
    -0.14, -0.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.36, 0.36, 0.16, 0.0, 0.16, -0.20, -0.20,
    0.16, 0.0, -0.16, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.10, 1.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
], dtype=torch.float32)


class TaksT1AmpEnv(DirectRLEnv):
    """Taks_T1 AMP环境 - 与S2S2R完全兼容
    
    观测空间（无特权观测，只用IMU+关节编码器）：
    - ang_vel: 3 (IMU角速度, body frame, scale=0.25)
    - gravity_vec: 3 (投影重力)
    - commands: 3 (速度命令)
    - dof_pos: 32 (关节位置, 训练顺序, scale=1.0)
    - dof_vel: 32 (关节速度, 训练顺序, scale=0.05)
    - actions: 32 (上一步动作)
    总计: 105维
    """
    
    cfg: TaksT1AmpEnvCfg

    def __init__(self, cfg: TaksT1AmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 构建URDF顺序到训练顺序的映射
        self.training_joint_indices = torch.tensor(
            [self.robot.data.joint_names.index(name) for name in TRAINING_JOINT_NAMES],
            device=self.device, dtype=torch.long
        )
        
        # 默认关节位置（训练顺序）
        self.default_dof_pos = DEFAULT_DOF_POS.to(self.device)
        
        # 动作缩放（与S2S2R一致）
        self.action_scale = 0.25

        # 加载motion数据
        self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device)

        # 关键body索引（用于AMP观测）
        key_body_names = ["left_wrist_pitch_link", "right_wrist_pitch_link", 
                          "left_ankle_roll_link", "right_ankle_roll_link"]
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        self.motion_dof_indexes = self._motion_loader.get_dof_index(TRAINING_JOINT_NAMES)
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(key_body_names)

        # AMP观测空间配置
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )

        # velocity command相关
        self.velocity_commands = torch.zeros((self.num_envs, 3), device=self.device)
        self.command_time_left = torch.zeros(self.num_envs, device=self.device)
        
        # 上一步动作缓存（32维，训练顺序）
        self.last_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        
        # 重力向量
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)

        # 随机推力计时器
        self.push_time_left = torch.zeros(self.num_envs, device=self.device)
        self._sample_push_interval(torch.arange(self.num_envs, device=self.device))

    def _sample_push_interval(self, env_ids: torch.Tensor):
        """重新采样推力间隔时间"""
        num_envs = len(env_ids)
        self.push_time_left[env_ids] = torch.empty(num_envs, device=self.device).uniform_(
            self.cfg.push_interval_s[0], self.cfg.push_interval_s[1]
        )

    def _apply_push(self):
        """应用随机推力 - GPU向量化计算"""
        if not self.cfg.enable_push:
            return
        
        # 检查哪些环境需要推力
        push_mask = self.push_time_left <= 0
        if not push_mask.any():
            return
        
        push_ids = torch.where(push_mask)[0]
        num_push = len(push_ids)
        
        # GPU向量化生成随机速度推力
        push_vel = torch.empty(num_push, 2, device=self.device).uniform_(
            self.cfg.push_vel_xy[0], self.cfg.push_vel_xy[1]
        )
        
        # 获取当前速度并添加推力
        root_vel = self.robot.data.root_com_vel_w[push_ids].clone()
        root_vel[:, :2] += push_vel
        self.robot.write_root_com_velocity_to_sim(root_vel, push_ids)
        
        # 重新采样推力间隔
        self._sample_push_interval(push_ids)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # 添加地面
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        # 克隆环境
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        self.scene.articulations["robot"] = self.robot
        # 添加灯光
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _randomize_friction(self, env_ids: torch.Tensor):
        """随机化摩擦力 - 优化版本（API限制需CPU，但减少重复计算）"""
        if not self.cfg.enable_friction_randomization:
            return
        
        # 转换为CPU tensor（API限制）
        env_ids_cpu = env_ids.cpu()
        num_envs = len(env_ids_cpu)
        total_num_shapes = self.robot.root_physx_view.max_shapes
        
        # 预计算范围差值（避免重复计算）
        sf_range = self.cfg.static_friction_range[1] - self.cfg.static_friction_range[0]
        df_range = self.cfg.dynamic_friction_range[1] - self.cfg.dynamic_friction_range[0]
        re_range = self.cfg.restitution_range[1] - self.cfg.restitution_range[0]
        
        # 使用torch.rand更高效，然后线性变换
        rand_vals = torch.rand(num_envs, total_num_shapes, 3, device="cpu")
        
        # 获取材质属性并更新
        materials = self.robot.root_physx_view.get_material_properties()
        materials[env_ids_cpu, :, 0] = rand_vals[:, :, 0] * sf_range + self.cfg.static_friction_range[0]
        materials[env_ids_cpu, :, 1] = rand_vals[:, :, 1] * df_range + self.cfg.dynamic_friction_range[0]
        materials[env_ids_cpu, :, 2] = rand_vals[:, :, 2] * re_range + self.cfg.restitution_range[0]
        self.robot.root_physx_view.set_material_properties(materials, env_ids_cpu)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        # 动作是训练顺序(32)，需要转换为URDF顺序后设置
        # target_training = default_pos + action_scale * actions
        target_training = self.default_dof_pos + self.action_scale * self.actions
        
        # 创建完整的关节目标（URDF顺序）
        target_urdf = self.robot.data.default_joint_pos.clone()
        target_urdf[:, self.training_joint_indices] = target_training
        self.robot.set_joint_position_target(target_urdf)

    def _get_observations(self) -> dict:
        # 计算投影重力（body frame）- 真机可从IMU获取
        root_quat = self.robot.data.root_quat_w
        projected_gravity = quat_apply_inverse(root_quat, self.gravity_vec)
        
        # 计算body frame下的角速度 - 真机可从IMU获取
        base_ang_vel = quat_apply_inverse(root_quat, self.robot.data.root_ang_vel_w)
        
        # 获取训练顺序的关节位置和速度（真机可从编码器获取）
        joint_pos_training = self.robot.data.joint_pos[:, self.training_joint_indices]
        joint_vel_training = self.robot.data.joint_vel[:, self.training_joint_indices]
        
        # 关节位置相对于默认位置
        joint_pos_rel = joint_pos_training - self.default_dof_pos
        
        # 添加观测噪声（GPU向量化计算）
        if self.cfg.enable_noise:
            # 使用torch.rand_like更高效，然后线性变换到目标范围
            base_ang_vel = base_ang_vel + (torch.rand_like(base_ang_vel) * (self.cfg.noise_ang_vel[1] - self.cfg.noise_ang_vel[0]) + self.cfg.noise_ang_vel[0])
            projected_gravity = projected_gravity + (torch.rand_like(projected_gravity) * (self.cfg.noise_gravity[1] - self.cfg.noise_gravity[0]) + self.cfg.noise_gravity[0])
            joint_pos_rel = joint_pos_rel + (torch.rand_like(joint_pos_rel) * (self.cfg.noise_joint_pos[1] - self.cfg.noise_joint_pos[0]) + self.cfg.noise_joint_pos[0])
            joint_vel_training = joint_vel_training + (torch.rand_like(joint_vel_training) * (self.cfg.noise_joint_vel[1] - self.cfg.noise_joint_vel[0]) + self.cfg.noise_joint_vel[0])
        
        # 构建策略观测（与S2S2R完全一致，105维）
        obs = torch.cat([
            base_ang_vel * 0.25,      # ang_vel, scale=0.25
            projected_gravity,         # gravity_vec
            self.velocity_commands,    # commands
            joint_pos_rel,             # dof_pos, scale=1.0
            joint_vel_training * 0.05, # dof_vel, scale=0.05
            self.last_actions,         # actions
        ], dim=-1)

        # 构建AMP观测（89维）
        amp_obs = compute_amp_obs(
            joint_pos_training,
            joint_vel_training,
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            self.robot.data.body_pos_w[:, self.key_body_indexes],
        )

        # 更新AMP观测历史
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        self.amp_observation_buffer[:, 0] = amp_obs.clone()
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}

        # 更新上一步动作
        self.last_actions = self.actions.clone()

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # 速度跟踪奖励
        lin_vel_error = torch.sum(
            torch.square(self.velocity_commands[:, :2] - self.robot.data.root_lin_vel_w[:, :2]), dim=1
        )
        ang_vel_error = torch.square(self.velocity_commands[:, 2] - self.robot.data.root_ang_vel_w[:, 2])
        
        # 使用指数奖励
        lin_vel_reward = torch.exp(-lin_vel_error / 0.25)
        ang_vel_reward = torch.exp(-ang_vel_error / 0.25)
        
        # 组合奖励（AMP的主要奖励来自判别器，这里只提供任务奖励）
        task_reward = 0.5 * lin_vel_reward + 0.5 * ang_vel_reward
        
        return task_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        # 确保env_ids是Tensor类型
        assert env_ids is not None
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # 重置velocity commands
        self._resample_commands(env_ids)
        
        # 重置上一步动作
        self.last_actions[env_ids] = 0.0
        
        # 重置推力计时器
        self._sample_push_interval(env_ids)
        
        # 随机化摩擦力
        self._randomize_friction(env_ids)

    def _resample_commands(self, env_ids: torch.Tensor):
        """重采样velocity commands"""
        num_envs = len(env_ids)
        
        # 采样新的速度命令
        self.velocity_commands[env_ids, 0] = torch.empty(num_envs, device=self.device).uniform_(
            self.cfg.lin_vel_x_range[0], self.cfg.lin_vel_x_range[1]
        )
        self.velocity_commands[env_ids, 1] = torch.empty(num_envs, device=self.device).uniform_(
            self.cfg.lin_vel_y_range[0], self.cfg.lin_vel_y_range[1]
        )
        self.velocity_commands[env_ids, 2] = torch.empty(num_envs, device=self.device).uniform_(
            self.cfg.ang_vel_z_range[0], self.cfg.ang_vel_z_range[1]
        )
        
        # 设置命令持续时间
        self.command_time_left[env_ids] = torch.empty(num_envs, device=self.device).uniform_(
            self.cfg.command_resampling_time[0], self.cfg.command_resampling_time[1]
        )
        
        # 10%的环境设置为静止命令
        standing_mask = torch.rand(num_envs, device=self.device) < 0.1
        self.velocity_commands[env_ids[standing_mask]] = 0.0

    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_samples = env_ids.shape[0]
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)
        
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)

        # 获取根body变换
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = body_positions[:, self.motion_ref_body_index] + self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.05
        root_state[:, 3:7] = body_rotations[:, self.motion_ref_body_index]
        root_state[:, 7:10] = body_linear_velocities[:, self.motion_ref_body_index]
        root_state[:, 10:13] = body_angular_velocities[:, self.motion_ref_body_index]
        
        # 获取训练顺序的关节状态
        dof_pos_training = dof_positions[:, self.motion_dof_indexes]
        dof_vel_training = dof_velocities[:, self.motion_dof_indexes]
        
        # 转换为URDF顺序的完整关节状态
        dof_pos = self.robot.data.default_joint_pos[env_ids].clone()
        dof_vel = self.robot.data.default_joint_vel[env_ids].clone()
        dof_pos[:, self.training_joint_indices] = dof_pos_training
        dof_vel[:, self.training_joint_indices] = dof_vel_training

        # 更新AMP观测
        amp_observations = self.collect_reference_motions(num_samples, times)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)

        return root_state, dof_pos, dof_vel

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        """收集参考motion数据用于AMP判别器"""
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader.dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()
        
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)
        
        amp_observation = compute_amp_obs(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
        )
        return amp_observation.view(-1, self.amp_observation_size)

    def step(self, action: torch.Tensor):
        """重写step以支持command重采样和随机推力"""
        obs, reward, terminated, truncated, info = super().step(action)
        
        dt = self.cfg.sim.dt * self.cfg.decimation
        
        # 更新命令时间
        self.command_time_left -= dt
        
        # 重采样超时的命令
        resample_ids = torch.where(self.command_time_left <= 0)[0]
        if len(resample_ids) > 0:
            self._resample_commands(resample_ids)
        
        # 更新推力时间并应用推力
        self.push_time_left -= dt
        self._apply_push()
        
        return obs, reward, terminated, truncated, info


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    """将四元数转换为切向量和法向量"""
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)


@torch.jit.script
def compute_amp_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
) -> torch.Tensor:
    """计算AMP观测（89维）
    
    返回: joint_pos(32) + joint_vel(32) + root_height(1) + root_orientation(6) + 
          root_lin_vel(3) + root_ang_vel(3) + key_body_pos(12) = 89维
    """
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],
            quaternion_to_tangent_and_normal(root_rotations),
            root_linear_velocities,
            root_angular_velocities,
            (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),
        ),
        dim=-1,
    )
    return obs
