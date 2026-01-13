# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Taks_T1 AMP环境实现，支持velocity command控制"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_rotate_inverse

from .taks_t1_amp_env_cfg import TaksT1AmpEnvCfg
from .motions import MotionLoader


class TaksT1AmpEnv(DirectRLEnv):
    """Taks_T1 AMP环境，支持velocity command控制
    
    观测空间与rough_env_cfg保持一致：
    - base_ang_vel: 3 (IMU角速度，body frame)
    - projected_gravity: 3 (投影重力)
    - velocity_commands: 3 (速度命令)
    - joint_pos: 35 (关节位置)
    - joint_vel: 35 (关节速度)
    - actions: 35 (上一步动作)
    """
    
    cfg: TaksT1AmpEnvCfg

    def __init__(self, cfg: TaksT1AmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 动作缩放
        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = self.robot.data.default_joint_pos[0].clone()
        self.action_scale = 0.25  # 与rough_env_cfg一致

        # 加载motion数据
        self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device)

        # 关键body索引（用于AMP观测）
        # Taks_T1的关键body: 左右手腕、左右脚踝
        key_body_names = ["left_wrist_pitch_link", "right_wrist_pitch_link", 
                          "left_ankle_roll_link", "right_ankle_roll_link"]
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(key_body_names)

        # AMP观测空间配置
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )

        # velocity command相关
        self.velocity_commands = torch.zeros((self.num_envs, 3), device=self.device)  # [vx, vy, wz]
        self.command_time_left = torch.zeros(self.num_envs, device=self.device)
        
        # 上一步动作缓存
        self.last_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        
        # 重力向量
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)

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

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        # 位置控制，与rough_env_cfg一致
        target = self.action_offset + self.action_scale * self.actions
        self.robot.set_joint_position_target(target)

    def _get_observations(self) -> dict:
        # 计算投影重力（body frame）
        root_quat = self.robot.data.root_quat_w
        projected_gravity = quat_rotate_inverse(root_quat, self.gravity_vec)
        
        # 计算body frame下的角速度
        base_ang_vel = quat_rotate_inverse(root_quat, self.robot.data.root_ang_vel_w)
        
        # 关节位置（相对于默认位置）
        joint_pos_rel = self.robot.data.joint_pos - self.robot.data.default_joint_pos
        
        # 关节速度
        joint_vel = self.robot.data.joint_vel
        
        # 构建策略观测（与rough_env_cfg一致）
        obs = torch.cat([
            base_ang_vel * 0.25,  # scale与rough_env_cfg一致
            projected_gravity,
            self.velocity_commands,  # velocity command
            joint_pos_rel,  # scale=1.0
            joint_vel * 0.05,  # scale与rough_env_cfg一致
            self.last_actions,
        ], dim=-1)

        # 构建AMP观测
        amp_obs = compute_amp_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
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
        root_state[:, 2] += 0.05  # 稍微抬高避免碰撞
        root_state[:, 3:7] = body_rotations[:, self.motion_ref_body_index]
        root_state[:, 7:10] = body_linear_velocities[:, self.motion_ref_body_index]
        root_state[:, 10:13] = body_angular_velocities[:, self.motion_ref_body_index]
        
        # 获取关节状态
        dof_pos = dof_positions[:, self.motion_dof_indexes]
        dof_vel = dof_velocities[:, self.motion_dof_indexes]

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
        """重写step以支持command重采样"""
        obs, reward, terminated, truncated, info = super().step(action)
        
        # 更新命令时间
        self.command_time_left -= self.cfg.sim.dt * self.cfg.decimation
        
        # 重采样超时的命令
        resample_ids = torch.where(self.command_time_left <= 0)[0]
        if len(resample_ids) > 0:
            self._resample_commands(resample_ids)
        
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
    """计算AMP观测
    
    返回: joint_pos(35) + joint_vel(35) + root_height(1) + root_orientation(6) + 
          root_lin_vel(3) + root_ang_vel(3) + key_body_pos(12) = 95维
    """
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],  # root body height
            quaternion_to_tangent_and_normal(root_rotations),
            root_linear_velocities,
            root_angular_velocities,
            (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),
        ),
        dim=-1,
    )
    return obs
