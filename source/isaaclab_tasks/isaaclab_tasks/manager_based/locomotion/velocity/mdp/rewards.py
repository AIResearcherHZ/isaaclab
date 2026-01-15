# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def gait_symmetry(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """奖励双脚步态对称性，鼓励左右脚接触时间平衡。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    # 计算左右脚接触时间差异，差异越小奖励越高
    time_diff = torch.abs(contact_time[:, 0] - contact_time[:, 1])
    return torch.exp(-time_diff * 10.0)  # 等价于 / 0.1


def double_support_time_penalty(
    env, sensor_cfg: SceneEntityCfg, max_double_support_time: float = 0.4
) -> torch.Tensor:
    """惩罚双脚同时接触地面时间过长，鼓励交替迈步。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    # 判断双脚是否同时接触地面
    both_in_contact = (contact_time[:, 0] > 0.0) & (contact_time[:, 1] > 0.0)
    # 以双脚接触时间的最小值作为双支撑时间，超出阈值则惩罚
    double_support_time = torch.min(contact_time, dim=1)[0]
    penalty = torch.clamp(double_support_time - max_double_support_time, min=0.0)
    return penalty * both_in_contact.float()


def base_height_reward(
    env, asset_cfg: SceneEntityCfg, target_height: float, tolerance: float = 0.1
) -> torch.Tensor:
    """奖励维持目标基座高度以防蹲下。"""
    asset = env.scene[asset_cfg.name]
    # 获取基座高度
    base_height = asset.data.root_pos_w[:, 2]
    # 计算高度误差
    height_error = torch.abs(base_height - target_height)
    # 使用指数核计算奖励
    reward = torch.exp(-height_error / tolerance)
    return reward


def knee_bend_penalty(
    env, asset_cfg: SceneEntityCfg, max_bend_angle: float = 0.78
) -> torch.Tensor:
    """惩罚膝关节过度弯曲以防止蹲姿。"""
    asset = env.scene[asset_cfg.name]
    # 获取膝关节位置
    knee_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    # 计算超出最大弯曲角度的部分
    excess_bend = torch.clamp(torch.abs(knee_pos) - max_bend_angle, min=0.0)
    # 返回所有膝关节的平均惩罚
    return torch.mean(excess_bend, dim=1)


def single_leg_stance_reward(
    env, sensor_cfg: SceneEntityCfg, command_name: str
) -> torch.Tensor:
    """奖励单脚支撑状态以鼓励正常迈步。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    # 单脚支撑 = 左右脚接触状态异或
    single_stance = (contact_time[:, 0] > 0.0) ^ (contact_time[:, 1] > 0.0)
    # 仅在运动命令存在时奖励
    command = env.command_manager.get_command(command_name)
    is_moving = torch.norm(command[:, :2], dim=1) > 0.1
    return single_stance.float() * is_moving.float()


def feet_alternating_contact(
    env, sensor_cfg: SceneEntityCfg, command_name: str
) -> torch.Tensor:
    """奖励双脚交替接触地面，鼓励一脚着地一脚离地的正常步态。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    # 理想状态是一脚在空中一脚着地（异或操作）
    left_in_air = air_time[:, 0] > 0.0
    right_in_air = air_time[:, 1] > 0.0
    alternating = left_in_air ^ right_in_air
    # 仅在运动命令存在时应用
    command = env.command_manager.get_command(command_name)
    is_moving = torch.norm(command[:, :2], dim=1) > 0.1
    return alternating.float() * is_moving.float()


def stand_still_posture(
    env, command_name: str, command_threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """当命令接近零时，奖励保持标准站立姿态。"""
    command = env.command_manager.get_command(command_name)
    is_standing = torch.norm(command[:, :2], dim=1) < command_threshold

    asset = env.scene[asset_cfg.name]
    # 获取关节位置
    joint_pos = asset.data.joint_pos
    default_pos = asset.data.default_joint_pos

    # 计算与默认位置的偏差
    deviation = torch.sum(torch.square(joint_pos - default_pos), dim=1)

    # 仅在静止时奖励保持默认姿态
    return torch.exp(-deviation * 0.5) * is_standing.float()


def velocity_direction_alignment(
    env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """奖励实际速度方向与命令方向对齐。"""
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    current_vel = vel_yaw[:, :2]
    command = env.command_manager.get_command(command_name)[:, :2]

    # 计算速度和命令的归一化点积
    current_speed = torch.norm(current_vel, dim=1)
    command_speed = torch.norm(command, dim=1)

    # 避免除零
    valid_mask = (current_speed > 0.1) & (command_speed > 0.1)

    dot_product = torch.sum(current_vel * command, dim=1)
    alignment = dot_product / (current_speed * command_speed + 1e-6)

    # 归一化到[0, 1]范围
    reward = (alignment + 1.0) / 2.0
    return reward * valid_mask.float()


def center_of_mass_stability(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    std: float = 0.1,
) -> torch.Tensor:
    """奖励重心在支撑区域内保持稳定。"""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 获取重心和双脚位置
    com_xy = asset.data.root_com_pos_w[:, :2]
    foot_pos_w = asset.data.body_pos_w[:, sensor_cfg.body_ids, :]
    # 获取接触状态
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = torch.norm(contact_forces[:, 0, :, :], dim=-1) > 1.0
    # 计算支撑中心
    contact_weights = in_contact.float()
    total_weight = contact_weights.sum(dim=1, keepdim=True).clamp(min=1.0)
    support_center = (foot_pos_w * contact_weights.unsqueeze(-1)).sum(dim=1) / total_weight
    # 计算距离并返回奖励
    distance = torch.norm(com_xy - support_center[:, :2], dim=1)
    return torch.exp(-distance / std)


def center_of_mass_velocity_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_velocity: float = 0.5,
) -> torch.Tensor:
    """惩罚重心水平速度过大，鼓励平稳运动。"""
    asset = env.scene[asset_cfg.name]
    horizontal_speed = torch.norm(asset.data.root_com_lin_vel_w[:, :2], dim=1)
    return torch.clamp(horizontal_speed - max_velocity, min=0.0)


def center_of_mass_height_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.8,
    tolerance: float = 0.1,
) -> torch.Tensor:
    """奖励重心高度保持在目标范围内。"""
    asset = env.scene[asset_cfg.name]
    height_error = torch.abs(asset.data.root_com_pos_w[:, 2] - target_height)
    return torch.exp(-height_error / tolerance)


def center_of_mass_in_support_polygon(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    margin: float = 0.05,
) -> torch.Tensor:
    """奖励重心投影在支撑多边形内部。
    
    对于双足机器人，支撑多边形由接触脚的位置定义。
    当只有一只脚接触时，奖励重心靠近该脚；
    当双脚接触时，奖励重心在两脚连线之间。
    
    Args:
        env: 环境对象
        asset_cfg: 资产配置
        sensor_cfg: 接触传感器配置
        margin: 安全边距 (m)
    
    Returns:
        重心在支撑区域内的奖励值
    """
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # 获取重心位置
    com_xy = asset.data.root_com_pos_w[:, :2]

    # 获取双脚位置
    foot_pos_w = asset.data.body_pos_w[:, sensor_cfg.body_ids, :]
    left_foot_xy = foot_pos_w[:, 0, :2]
    right_foot_xy = foot_pos_w[:, 1, :2]

    # 获取接触状态
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = torch.norm(contact_forces[:, 0, :, :], dim=-1) > 1.0
    num_contacts = in_contact.sum(dim=1)

    # 双脚支撑奖励（纯张量操作，无.any()调用）
    foot_vec = right_foot_xy - left_foot_xy
    foot_dist = torch.norm(foot_vec, dim=1, keepdim=True).clamp(min=1e-6)
    foot_dir = foot_vec / foot_dist
    com_to_left = com_xy - left_foot_xy
    proj_length = (com_to_left * foot_dir).sum(dim=1)
    normalized_pos = proj_length / foot_dist.squeeze()
    center_dist = torch.abs(normalized_pos - 0.5)
    double_support_reward = torch.exp(-center_dist * 4.0)

    # 单脚支撑奖励（纯张量操作）
    support_foot_xy = torch.where(
        in_contact[:, 0:1].expand(-1, 2),
        left_foot_xy,
        right_foot_xy
    )
    dist_to_support = torch.norm(com_xy - support_foot_xy, dim=1)
    single_support_reward = torch.exp(-dist_to_support * 10.0)  # 等价于 / 0.1

    # 根据接触数量选择奖励
    both_contact = num_contacts == 2
    single_contact = num_contacts == 1
    reward = torch.where(both_contact, double_support_reward,
                         torch.where(single_contact, single_support_reward,
                                     torch.zeros_like(double_support_reward)))
    return reward


class FeetJitterPenalty:
    """脚部抖动惩罚类，预分配缓冲区避免hasattr检查。"""
    
    def __init__(self, env, asset_cfg: SceneEntityCfg):
        self.asset = env.scene[asset_cfg.name]
        self.body_ids = asset_cfg.body_ids
        num_envs = env.num_envs
        num_feet = len(self.body_ids) if isinstance(self.body_ids, list) else 2
        device = self.asset.device
        # 预分配历史缓冲区
        self._prev_foot_vel = torch.zeros(num_envs, num_feet, 3, device=device)
        self._prev_prev_foot_vel = torch.zeros(num_envs, num_feet, 3, device=device)
    
    def __call__(self) -> torch.Tensor:
        """计算脚部抖动惩罚。"""
        current_foot_vel = self.asset.data.body_lin_vel_w[:, self.body_ids, :]
        # 计算jerk（三阶导数）
        foot_acc = current_foot_vel - self._prev_foot_vel
        prev_foot_acc = self._prev_foot_vel - self._prev_prev_foot_vel
        foot_jerk = foot_acc - prev_foot_acc
        # L2惩罚
        jitter_penalty = torch.sum(torch.sum(foot_jerk**2, dim=-1), dim=-1)
        # 更新缓冲区（in-place copy避免clone开销）
        self._prev_prev_foot_vel.copy_(self._prev_foot_vel)
        self._prev_foot_vel.copy_(current_foot_vel)
        return jitter_penalty


# 全局实例缓存，避免每次调用重新创建
_feet_jitter_instances: dict = {}


def feet_jitter_penalty(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """惩罚脚部抖动：通过计算脚部速度的变化率来检测不必要的抖动。
    
    优化：使用类实例缓存和预分配缓冲区，避免hasattr检查和动态属性创建。
    """
    key = id(env)
    if key not in _feet_jitter_instances:
        _feet_jitter_instances[key] = FeetJitterPenalty(env, asset_cfg)
    return _feet_jitter_instances[key]()

    
# ==================== 条件奖励函数（根据指令状态切换） ====================

# 指令掩码缓存，避免同一step内多次计算
_cmd_mask_cache: dict = {}


def _get_command_mask(env, command_name: str, threshold: float = 0.1) -> torch.Tensor:
    """获取指令掩码：有指令时为1，无指令时为0。
    
    优化：同一step内缓存结果，避免多个条件奖励函数重复计算。
    """
    cache_key = (id(env), env.common_step_counter, command_name, threshold)
    if cache_key in _cmd_mask_cache:
        return _cmd_mask_cache[cache_key]
    
    command = env.command_manager.get_command(command_name)
    # 计算指令幅度（线速度xy + 角速度z）
    cmd_magnitude = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    mask = (cmd_magnitude > threshold).float()
    
    # 缓存结果（只保留当前step的缓存）
    _cmd_mask_cache.clear()  # 清空旧缓存
    _cmd_mask_cache[cache_key] = mask
    return mask


def action_rate_l2_conditional(
    env, command_name: str, command_threshold: float = 0.1
) -> torch.Tensor:
    """条件动作变化率惩罚：仅在有指令时应用。
    
    无指令时不惩罚动作变化，允许机器人自由调整以保持平衡。
    """
    action_rate = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    cmd_mask = _get_command_mask(env, command_name, command_threshold)
    return action_rate * cmd_mask


def dof_acc_l2_conditional(
    env, command_name: str, command_threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """条件关节加速度惩罚：仅在有指令时应用。
    
    无指令时不惩罚加速度，允许机器人快速响应外部扰动。
    """
    asset = env.scene[asset_cfg.name]
    acc = torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)
    cmd_mask = _get_command_mask(env, command_name, command_threshold)
    return acc * cmd_mask


def dof_torques_l2_conditional(
    env, command_name: str, command_threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """条件关节扭矩惩罚：仅在有指令时应用。
    
    无指令时不惩罚扭矩，允许机器人使用必要的扭矩来抵抗外部干扰。
    """
    asset = env.scene[asset_cfg.name]
    torques = torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
    cmd_mask = _get_command_mask(env, command_name, command_threshold)
    return torques * cmd_mask


def joint_torques_l2_conditional(
    env, command_name: str, command_threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """条件关节扭矩L2惩罚：仅在有指令时应用。
    
    与dof_torques_l2_conditional相同，用于特定关节组。
    """
    return dof_torques_l2_conditional(env, command_name, command_threshold, asset_cfg)


def gait_symmetry_conditional(
    env, command_name: str, command_threshold: float = 0.1, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")
) -> torch.Tensor:
    """条件步态对称性奖励：仅在有指令时应用。"""
    reward = gait_symmetry(env, sensor_cfg)
    cmd_mask = _get_command_mask(env, command_name, command_threshold)
    return reward * cmd_mask


def double_support_time_penalty_conditional(
    env, command_name: str, command_threshold: float = 0.1,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    max_double_support_time: float = 0.4
) -> torch.Tensor:
    """条件双脚支撑时间惩罚：仅在有指令时应用。"""
    reward = double_support_time_penalty(env, sensor_cfg, max_double_support_time)
    cmd_mask = _get_command_mask(env, command_name, command_threshold)
    return reward * cmd_mask


def single_leg_stance_reward_conditional(
    env, command_name: str, command_threshold: float = 0.1,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")
) -> torch.Tensor:
    """条件单脚支撑奖励：仅在有指令时应用。"""
    reward = single_leg_stance_reward(env, sensor_cfg, command_name)
    cmd_mask = _get_command_mask(env, command_name, command_threshold)
    return reward * cmd_mask


def feet_alternating_contact_conditional(
    env, command_name: str, command_threshold: float = 0.1,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")
) -> torch.Tensor:
    """条件双脚交替接触奖励：仅在有指令时应用。"""
    reward = feet_alternating_contact(env, sensor_cfg, command_name)
    cmd_mask = _get_command_mask(env, command_name, command_threshold)
    return reward * cmd_mask


def velocity_direction_alignment_conditional(
    env, command_name: str, command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """条件速度方向对齐奖励：仅在有指令时应用。"""
    reward = velocity_direction_alignment(env, command_name, asset_cfg)
    cmd_mask = _get_command_mask(env, command_name, command_threshold)
    return reward * cmd_mask


def feet_jitter_penalty_conditional(
    env, command_name: str, command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """条件脚部抖动惩罚：仅在有指令时应用。
    
    在静止状态下允许一定的调整动作来保持平衡，
    只在运动时惩罚不必要的脚部抖动。
    """
    penalty = feet_jitter_penalty(env, asset_cfg)
    cmd_mask = _get_command_mask(env, command_name, command_threshold)
    return penalty * cmd_mask