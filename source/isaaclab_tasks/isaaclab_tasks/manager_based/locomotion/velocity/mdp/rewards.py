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


def body_pitch_range_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_pitch: float = -0.3,
    max_pitch: float = 0.3,
    use_body_link: bool = False,
) -> torch.Tensor:
    """惩罚pitch角度超出指定范围（支持前倾和后仰分别限制）。
    
    Args:
        env: 环境对象
        asset_cfg: 资产配置，可指定body_names来选择特定link
        min_pitch: 最小允许pitch角度（负值表示后仰），单位弧度
        max_pitch: 最大允许pitch角度（正值表示前倾），单位弧度
        use_body_link: 是否使用指定body link的姿态，False则使用root姿态
    
    Returns:
        超出范围的惩罚值（越界越大）
    """
    asset = env.scene[asset_cfg.name]
    
    if use_body_link and asset_cfg.body_ids is not None:
        # 使用指定body link的世界姿态
        quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    else:
        # 使用机器人根节点姿态
        quat = asset.data.root_quat_w
    
    # 计算pitch角度 (正值=前倾, 负值=后仰)
    pitch = torch.asin(torch.clamp(2.0 * (quat[:, 0] * quat[:, 2] - quat[:, 3] * quat[:, 1]), -1.0, 1.0))
    
    # 分别计算前倾和后仰的越界量
    excess_forward = torch.clamp(pitch - max_pitch, min=0.0)  # 前倾超限
    excess_backward = torch.clamp(min_pitch - pitch, min=0.0)  # 后仰超限
    
    return excess_forward + excess_backward


def foot_roll_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    max_roll: float = 0.15,
) -> torch.Tensor:
    """惩罚脚踝roll角度过大（防止脚侧面着地）。
    
    只在脚接触地面时惩罚，鼓励全脚掌着地。
    
    Args:
        env: 环境对象
        asset_cfg: 资产配置，指定脚踝link
        sensor_cfg: 接触传感器配置
        max_roll: 最大允许roll角度，单位弧度（默认约8.6度）
    
    Returns:
        脚踝roll角度超限的惩罚值
    """
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取脚踝link的世界姿态四元数
    foot_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]  # (num_envs, num_feet, 4)
    
    # 计算roll角度: roll = atan2(2*(w*x + y*z), 1 - 2*(x^2 + y^2))
    w, x, y, z = foot_quat[..., 0], foot_quat[..., 1], foot_quat[..., 2], foot_quat[..., 3]
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    
    # 获取接触状态
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = torch.norm(contact_forces[:, 0, :, :], dim=-1) > 1.0  # (num_envs, num_feet)
    
    # 只在接触时惩罚roll角度超限
    excess_roll = torch.clamp(torch.abs(roll) - max_roll, min=0.0)  # (num_envs, num_feet)
    penalty = excess_roll * in_contact.float()
    
    return torch.sum(penalty, dim=1)


def foot_flat_contact_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    ideal_roll: float = 0.0,
    ideal_pitch: float = 0.0,
    roll_tolerance: float = 0.02,
    pitch_tolerance: float = 0.10,
) -> torch.Tensor:
    """奖励脚全掌平稳着地（roll和pitch都接近理想值）。
    
    Args:
        env: 环境对象
        asset_cfg: 资产配置，指定脚踝link
        sensor_cfg: 接触传感器配置
        ideal_roll: 理想roll角度，默认0
        ideal_pitch: 理想pitch角度，默认0
        roll_tolerance: roll容差
        pitch_tolerance: pitch容差
    
    Returns:
        脚平稳着地的奖励值
    """
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取脚踝link的世界姿态四元数
    foot_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]  # (num_envs, num_feet, 4)
    
    w, x, y, z = foot_quat[..., 0], foot_quat[..., 1], foot_quat[..., 2], foot_quat[..., 3]
    
    # 计算roll和pitch
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    
    # 获取接触状态
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = torch.norm(contact_forces[:, 0, :, :], dim=-1) > 1.0  # (num_envs, num_feet)
    
    # 计算与理想姿态的偏差
    roll_error = torch.abs(roll - ideal_roll)
    pitch_error = torch.abs(pitch - ideal_pitch)
    
    # 在容差范围内给予奖励
    roll_reward = torch.exp(-roll_error / roll_tolerance)
    pitch_reward = torch.exp(-pitch_error / pitch_tolerance)
    
    # 只在接触时奖励
    reward = roll_reward * pitch_reward * in_contact.float()
    
    return torch.sum(reward, dim=1)