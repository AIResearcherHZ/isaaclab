# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Semi-Taks-T1半身机器人Reach环境配置（关节位置控制）。"""

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.config.semi_taks_t1.reach_env_cfg import ReachEnvCfg

from isaaclab_assets.robots.taks import SEMI_TAKS_T1_CFG


@configclass
class SemiTaksT1ReachEnvCfg(ReachEnvCfg):
    """Semi-Taks-T1半身机器人Reach环境配置。
    
    20 DOF结构（17 DOF参与训练）：
    - 双臂14 DOF（每臂7 DOF）- 参与训练
    - 腰部3 DOF - 参与训练，用于补偿
    - 颈部3 DOF - 锁定，不参与训练
    
    动作空间（17维）：
    - left_arm_action: 7维
    - right_arm_action: 7维  
    - waist_action: 3维
    
    观测空间（62维）：
    - 关节位置: 17维（双臂14 + 腰部3）
    - 关节速度: 17维
    - 目标位姿: 14维（左右各7维）
    - 上一步动作: 14维（左右臂各7维）
    """

    def __post_init__(self):
        super().__post_init__()

        # 使用Semi-Taks-T1机器人
        self.scene.robot = SEMI_TAKS_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 末端执行器body名称（双臂pitch末端）
        self.rewards.left_end_effector_position_tracking.params["asset_cfg"].body_names = ["left_wrist_pitch_link"]
        self.rewards.left_end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [
            "left_wrist_pitch_link"
        ]
        self.rewards.left_end_effector_orientation_tracking.params["asset_cfg"].body_names = ["left_wrist_pitch_link"]

        self.rewards.right_end_effector_position_tracking.params["asset_cfg"].body_names = ["right_wrist_pitch_link"]
        self.rewards.right_end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [
            "right_wrist_pitch_link"
        ]
        self.rewards.right_end_effector_orientation_tracking.params["asset_cfg"].body_names = ["right_wrist_pitch_link"]

        # 左臂动作 (7 DOF)
        self.actions.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint", "left_elbow_joint",
                "left_wrist_roll_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint",
            ],
            scale=0.5,
            use_default_offset=True,
        )

        # 右臂动作 (7 DOF)
        self.actions.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint", "right_elbow_joint",
                "right_wrist_roll_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint",
            ],
            scale=0.5,
            use_default_offset=True,
        )

        # 腰部动作 (3 DOF) - 较小scale使运动更平滑，作为补偿使用
        self.actions.waist_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
            scale=0.2,
            use_default_offset=True,
        )

        # 目标追踪body（双臂末端pitch link）
        self.commands.left_ee_pose.body_name = "left_wrist_pitch_link"
        self.commands.right_ee_pose.body_name = "right_wrist_pitch_link"


@configclass
class SemiTaksT1ReachEnvCfg_PLAY(SemiTaksT1ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 播放时使用较少环境
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # 禁用随机化
        self.observations.policy.enable_corruption = False
