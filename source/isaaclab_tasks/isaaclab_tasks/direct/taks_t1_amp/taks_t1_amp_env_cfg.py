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
    
    观测空间与S2S2R完全一致（无特权观测，只用IMU+关节编码器）：
    - ang_vel: 3 (IMU角速度, body frame)
    - gravity_vec: 3 (投影重力)
    - commands: 3 (速度命令)
    - dof_pos: 32 (关节位置, 训练顺序)
    - dof_vel: 32 (关节速度, 训练顺序)
    - actions: 32 (上一步动作)
    总计: 105维观测
    
    AMP观测空间（用于判别器）：
    - joint_pos: 32
    - joint_vel: 32
    - root_height: 1
    - root_orientation: 6 (tangent + normal)
    - root_lin_vel: 3
    - root_ang_vel: 3
    - key_body_pos: 12 (4个关键body * 3)
    总计: 89维AMP观测
    """

    # 环境参数
    episode_length_s = 10.0
    decimation = 4

    # 观测空间（与S2S2R完全一致）
    # ang_vel(3) + gravity_vec(3) + commands(3) + dof_pos(32) + dof_vel(32) + actions(32) = 105
    observation_space = 105
    action_space = 32
    state_space = 0
    
    # AMP观测空间
    num_amp_observations = 2
    # joint_pos(32) + joint_vel(32) + root_height(1) + root_orientation(6) + root_lin_vel(3) + root_ang_vel(3) + key_body_pos(12) = 89
    amp_observation_space = 89

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

@configclass
class TaksT1AmpDanceEnvCfg(TaksT1AmpEnvCfg):
    """Taks_T1 AMP Dance环境配置"""
    motion_file = os.path.join(MOTIONS_DIR, "taks_t1_dance.npz")