# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""用于启用终止条件的通用函数。

这些函数用于在满足条件时提前结束回合（例如：走出地形边界）。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """当智能体距离地形边缘过近时触发终止。

    参数：
        安全边界距离：数值越大则越早触发终止。
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        # 平面地形可视为“无限大”，不存在越界问题，因此永远不终止。
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # 生成式地形由多个子地形拼成，需要根据配置计算整张地图尺寸。
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # 地图总宽高 = 行列数 * 单块尺寸 + 两侧边框。
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # 显式声明类型，方便类型提示与代码跳转。
        asset: RigidObject = env.scene[asset_cfg.name]

        # 如果根部位置在任一水平轴方向接近地图边界（扣除安全边界距离），则认为越界。
        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        # 任一方向越界即终止。
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        # 仅支持两种内置地形类型。
        raise ValueError("收到不支持的地形类型，只支持两种内置类型。")
