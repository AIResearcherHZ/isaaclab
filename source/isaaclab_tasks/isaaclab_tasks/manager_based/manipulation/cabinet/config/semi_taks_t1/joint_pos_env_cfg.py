# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##
# Pre-defined configs
##
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_assets.robots.taks import SEMI_TAKS_T1_CFG

from isaaclab_tasks.manager_based.manipulation.cabinet.config.openarm.cabinet_openarm_env_cfg import (  # isort: skip
    FRAME_MARKER_SMALL_CFG,
    CabinetEnvCfg,
)


@configclass
class SemiTaksT1CabinetEnvCfg(CabinetEnvCfg):
    """Semi-Taks-T1半身机器人Cabinet环境配置。
    
    20 DOF结构（10 DOF参与训练）：
    - 右臂7 DOF - 参与训练（主要操作臂）
    - 腰部3 DOF - 参与训练，用于补偿
    - 左臂7 DOF - 不参与训练
    - 颈部3 DOF - 锁定，不参与训练
    
    动作空间（10维）：
    - arm_action: 7维（右臂）
    - waist_action: 3维
    """

    def __post_init__(self):
        super().__post_init__()

        # 使用Semi-Taks-T1机器人
        self.scene.robot = SEMI_TAKS_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 右臂动作 (7 DOF)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint", "right_elbow_joint",
                "right_wrist_roll_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint",
            ],
            scale=0.5,
            use_default_offset=True,
        )

        # 腰部动作 (3 DOF) - 较小scale使运动更平滑
        self.actions.waist_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
            scale=0.2,
            use_default_offset=True,
        )

        # 无夹爪，使用空配置
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["right_wrist_pitch_joint"],
            open_command_expr={"right_wrist_pitch_joint": 0.0},
            close_command_expr={"right_wrist_pitch_joint": 0.0},
        )

        # 末端执行器帧配置
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_wrist_pitch_link",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, -0.003),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_wrist_pitch_link",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, -0.005, 0.075),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_wrist_pitch_link",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.005, 0.075),
                    ),
                ),
            ],
        )

        # 奖励配置
        self.rewards.approach_gripper_handle.params["offset"] = 0.04
        self.rewards.grasp_handle.params["open_joint_pos"] = 0.0
        self.rewards.grasp_handle.params["asset_cfg"].joint_names = ["right_wrist_pitch_joint"]


@configclass
class SemiTaksT1CabinetEnvCfg_PLAY(SemiTaksT1CabinetEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
