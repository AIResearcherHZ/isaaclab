# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import math

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.config.openarm.lift_openarm_env_cfg import LiftEnvCfg

from isaaclab_assets.robots.taks import SEMI_TAKS_T1_CFG

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


@configclass
class SemiTaksT1CubeLiftEnvCfg(LiftEnvCfg):
    """Semi-Taks-T1半身机器人Lift环境配置。
    
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
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_yaw_joint",
                "right_wrist_pitch_joint",
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

        # 末端执行器body名称
        self.commands.object_pose.body_name = "right_wrist_pitch_link"
        self.commands.object_pose.ranges.pitch = (math.pi / 2, math.pi / 2)

        # 观测关节名称（右臂 + 腰部）
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = [
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint",
            "right_wrist_roll_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        ]
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = [
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint",
            "right_wrist_roll_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        ]

        # 奖励关节名称（右臂 + 腰部）
        self.rewards.joint_vel.params["asset_cfg"].joint_names = [
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint",
            "right_wrist_roll_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        ]

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_wrist_pitch_link",
                    name="end_effector",
                ),
            ],
        )


@configclass
class SemiTaksT1CubeLiftEnvCfg_PLAY(SemiTaksT1CubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
