# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Taks_T1 AMP环境配置，支持velocity command控制"""

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab_assets.robots.taks import TAKS_T1_CFG

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class TaksT1AmpEnvCfg(DirectRLEnvCfg):
    """Taks_T1 AMP环境配置基类
    
    观测空间与rough_env_cfg保持一致：
    - base_ang_vel: 3 (IMU角速度)
    - projected_gravity: 3 (投影重力)
    - velocity_commands: 3 (速度命令)
    - joint_pos: 35 (关节位置)
    - joint_vel: 35 (关节速度)
    - actions: 35 (上一步动作)
    总计: 114维观测
    
    AMP观测空间（用于判别器）：
    - joint_pos: 35
    - joint_vel: 35
    - root_height: 1
    - root_orientation: 6 (tangent + normal)
    - root_lin_vel: 3
    - root_ang_vel: 3
    - key_body_positions: 12 (4个关键body * 3)
    总计: 95维AMP观测
    """

    # 环境参数
    episode_length_s = 10.0
    decimation = 4

    # 观测空间（与rough_env_cfg一致）
    # base_ang_vel(3) + projected_gravity(3) + velocity_commands(3) + joint_pos(35) + joint_vel(35) + actions(35)
    observation_space = 114
    action_space = 35
    state_space = 0
    
    # AMP观测空间
    num_amp_observations = 2
    # joint_pos(35) + joint_vel(35) + root_height(1) + root_orientation(6) + root_lin_vel(3) + root_ang_vel(3) + key_body_pos(12)
    amp_observation_space = 95

    # 终止条件
    early_termination = True
    termination_height = 0.3

    # motion文件路径
    motion_file: str = MISSING
    reference_body = "pelvis"  # Taks_T1的根body
    reset_strategy = "random"  # default, random, random-start

    # 速度命令范围
    lin_vel_x_range = (-1.0, 1.0)
    lin_vel_y_range = (-1.0, 1.0)
    ang_vel_z_range = (-1.0, 1.0)
    command_resampling_time = (2.0, 10.0)

    # 仿真配置
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,  # 与rough_env_cfg一致: 0.005s
        render_interval=4,  # decimation
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
            enable_external_forces_every_iteration=True,
        ),
    )

    # 场景配置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=5.0, replicate_physics=True)

    # 机器人配置
    robot: ArticulationCfg = TAKS_T1_CFG.replace(prim_path="/World/envs/env_.*/Robot")


@configclass
class TaksT1AmpWalkEnvCfg(TaksT1AmpEnvCfg):
    """Taks_T1 AMP Walk环境配置"""
    motion_file = os.path.join(MOTIONS_DIR, "taks_t1_walk.npz")


@configclass
class TaksT1AmpRunEnvCfg(TaksT1AmpEnvCfg):
    """Taks_T1 AMP Run环境配置"""
    motion_file = os.path.join(MOTIONS_DIR, "taks_t1_run.npz")
