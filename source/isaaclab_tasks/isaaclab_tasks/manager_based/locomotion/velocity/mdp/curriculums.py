# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""用于构建学习环境的课程学习通用函数。

这些函数用于根据训练表现动态调整环境难度（例如地形等级）。
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """基于“机器人在期望速度指令下实际走了多远”来调节地形难度。

    规则：
        - 如果机器人走得足够远：提升到更难地形。
        - 如果机器人走得不足（小于指令期望距离的一半）：降低到更简单地形。

    说明：
        该规则只适用于可按等级逐步提升难度的生成式地形。

    返回：
        当前所有环境的地形等级均值。
    """
    # 显式声明类型，方便类型提示与代码跳转。
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # 计算机器人相对环境原点位置的水平位移距离。
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # 走到足够远则升级地形（阈值为子地形尺寸的一半）。
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # 若走的距离小于“期望速度 * 回合时长”的一半，则降低地形难度。
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    # 若已经升级，就不要同时降级（避免冲突）。
    move_down *= ~move_up
    # 根据升级/降级标记更新每个环境对应的地形等级与原点位置。
    terrain.update_env_origins(env_ids, move_up, move_down)
    # 返回全体环境的地形等级均值（可用于日志记录或监控课程进度）。
    return torch.mean(terrain.terrain_levels.float())