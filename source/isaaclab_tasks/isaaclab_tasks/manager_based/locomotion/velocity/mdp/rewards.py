"""可用于为学习环境定义奖励的通用函数。

这些函数可传递给 :class:`isaaclab.managers.RewardTermCfg` 对象以指定奖励函数及其参数。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """奖励脚步在空中保持较长时间以体现迈步动作。

    本函数奖励当脚步悬空时间超过阈值时的动作，帮助机器人抬脚并迈步。
    在命令较小时(即无需迈步)，将不予奖励。
    """
    # 提取接触传感器对象以便后续类型提示
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 计算首次接触以及当前空中时间
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # 若命令接近零则不奖励
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """奖励双足在空中轮流支撑并实现合理迈步。

    本函数鼓励双足中始终保持一只脚在空中，且空中时间不过度，起到步态规范的作用。
    命令较小时不予奖励。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 提取空中时间与接触时间
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # 仅在运动命令存在时奖励
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """惩罚脚部在接触地面时的滑移。

    本函数通过脚部线速度与接触状态的组合惩罚滑移，避免足端在地面上打滑。
    """
    # 提取接触传感器数据
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """使用指数核在重力对齐的机器人坐标系中奖励线速度(xy方向)追踪。"""
    # 提取所需资产数据
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """使用指数核在世界坐标系中奖励角速度(z轴)追踪。"""
    # 提取所需资产数据
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """当命令接近零时惩罚关节偏离默认位置的行为。"""
    command = env.command_manager.get_command(command_name)
    # 命令接近零时才会惩罚关节偏移
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def gait_symmetry(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """奖励双脚步态对称性，鼓励左右脚接触时间平衡。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    # 假设 body_ids[0] 为左脚，body_ids[1] 为右脚
    if contact_time.shape[1] >= 2:
        left_contact = contact_time[:, 0]
        right_contact = contact_time[:, 1]
        # 计算左右脚接触时间差异，差异越小奖励越高
        time_diff = torch.abs(left_contact - right_contact)
        reward = torch.exp(-time_diff / 0.1)
        return reward
    return torch.zeros(env.num_envs, device=env.device)

def double_support_time_penalty(
    env, sensor_cfg: SceneEntityCfg, max_double_support_time: float = 0.4
) -> torch.Tensor:
    """惩罚双脚同时接触地面时间过长，鼓励交替迈步。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    if contact_time.shape[1] >= 2:
        # 判断双脚是否同时接触地面
        both_in_contact = (contact_time[:, 0] > 0.0) & (contact_time[:, 1] > 0.0)
        # 以双脚接触时间的最小值作为双支撑时间
        double_support_time = torch.min(contact_time, dim=1)[0]
        # 超出阈值则惩罚
        penalty = torch.clamp(double_support_time - max_double_support_time, min=0.0)
        return penalty * both_in_contact.float()
    return torch.zeros(env.num_envs, device=env.device)


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

    if contact_time.shape[1] >= 2:
        left_contact = contact_time[:, 0] > 0.0
        right_contact = contact_time[:, 1] > 0.0
        single_stance = left_contact ^ right_contact

        # 仅在运动命令存在时奖励
        command = env.command_manager.get_command(command_name)
        is_moving = torch.norm(command[:, :2], dim=1) > 0.1

        return single_stance.float() * is_moving.float()
    return torch.zeros(env.num_envs, device=env.device)


def feet_alternating_contact(
    env, sensor_cfg: SceneEntityCfg, command_name: str
) -> torch.Tensor:
    """奖励双脚交替接触地面，鼓励一脚着地一脚离地的正常步态。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    if air_time.shape[1] >= 2:
        # 判断双脚当前状态
        left_in_air = air_time[:, 0] > 0.0
        right_in_air = air_time[:, 1] > 0.0

        both_in_air = left_in_air & right_in_air
        both_on_ground = (~left_in_air) & (~right_in_air)

        # 理想状态是一脚在空中一脚着地
        alternating = ~(both_in_air | both_on_ground)

        # 仅在运动命令存在时应用
        command = env.command_manager.get_command(command_name)
        is_moving = torch.norm(command[:, :2], dim=1) > 0.1

        return alternating.float() * is_moving.float()
    return torch.zeros(env.num_envs, device=env.device)


def command_direction_change_penalty(
    env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """当命令方向与当前运动方向相反时，惩罚过快的动作变化。"""
    asset = env.scene[asset_cfg.name]
    # 获取当前速度
    current_vel = asset.data.root_lin_vel_w[:, :2]
    # 获取目标命令
    command = env.command_manager.get_command(command_name)[:, :2]

    # 计算速度和命令的点积，负值表示方向相反
    dot_product = torch.sum(current_vel * command, dim=1)

    # 当方向相反且速度较大时给予惩罚
    current_speed = torch.norm(current_vel, dim=1)
    direction_change = (dot_product < 0) & (current_speed > 0.2)

    # 惩罚值与速度成正比
    return direction_change.float() * current_speed


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


def backward_walking_stability(
    env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """后退行走时奖励身体稳定性，鼓励适当的后仰姿态。"""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # 判断是否在后退（命令x为负）
    is_backward = command[:, 0] < -0.1
    # 获取躯干pitch角度（从四元数提取）
    quat = asset.data.root_quat_w
    # 计算pitch角度：arcsin(2*(w*y - z*x))
    pitch = torch.asin(torch.clamp(2.0 * (quat[:, 0] * quat[:, 2] - quat[:, 3] * quat[:, 1]), -1.0, 1.0))
    # 后退时允许轻微后仰（pitch为正），但不要过度
    # 理想pitch范围：0到0.1弧度（约0-6度）
    ideal_backward_pitch = 0.05
    pitch_error = torch.abs(pitch - ideal_backward_pitch)
    pitch_reward = torch.exp(-pitch_error * 5.0)
    return pitch_reward * is_backward.float()


def body_pitch_penalty(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), max_pitch: float = 0.3
) -> torch.Tensor:
    """惩罚躯干过度前倾或后仰。"""
    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    # 计算pitch角度
    pitch = torch.asin(torch.clamp(2.0 * (quat[:, 0] * quat[:, 2] - quat[:, 3] * quat[:, 1]), -1.0, 1.0))
    # 超出最大角度的部分给予惩罚
    excess_pitch = torch.clamp(torch.abs(pitch) - max_pitch, min=0.0)
    return excess_pitch