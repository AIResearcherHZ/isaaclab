# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##
# Pre-defined configs
##

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.config.semi_taks_t1.bimanual.reach_semi_taks_t1_bi_env_cfg import ReachEnvCfg

from isaaclab_assets.robots.taks import SEMI_TAKS_T1_HIGH_PD_CFG

##
# Environment configuration
##


@configclass
class SemiTaksT1ReachEnvCfg(ReachEnvCfg):
    """Configuration for the Bimanual Semi-Taks-T1 Reach Environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to Semi-Taks-T1
        self.scene.robot = SEMI_TAKS_T1_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # override rewards
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

        # override actions
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

        # override command generator body
        self.commands.left_ee_pose.body_name = "left_wrist_pitch_link"
        self.commands.right_ee_pose.body_name = "right_wrist_pitch_link"


@configclass
class SemiTaksT1ReachEnvCfg_PLAY(SemiTaksT1ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
