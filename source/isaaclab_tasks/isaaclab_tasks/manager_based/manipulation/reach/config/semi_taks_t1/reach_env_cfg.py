# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Semi-Taks-T1 半身机器人reach环境配置。

20 DOF结构：
- 双臂14 DOF（每臂7 DOF：shoulder_pitch/roll/yaw + elbow + wrist_roll/yaw/pitch）
- 腰部3 DOF（waist_yaw/roll/pitch）- 参与训练，用于补偿
- 颈部3 DOF（neck_yaw/roll/pitch）- 锁定，不参与训练

目标追踪：双臂末端(wrist_pitch_link)追踪目标点
控制策略：优先使用手臂，腰部补偿，避免剧烈变化
"""

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp

##
# Scene definition
##


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0)),
    )

    # robots
    robot: ArticulationCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    left_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.3),
            pos_y=(0.15, 0.25),
            pos_z=(0.3, 0.5),
            roll=(-math.pi / 6, math.pi / 6),
            pitch=(3 * math.pi / 2, 3 * math.pi / 2),
            yaw=(8 * math.pi / 9, 10 * math.pi / 9),
        ),
    )

    right_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.3),
            pos_y=(-0.25, -0.15),
            pos_z=(0.3, 0.5),
            roll=(-math.pi / 6, math.pi / 6),
            pitch=(3 * math.pi / 2, 3 * math.pi / 2),
            yaw=(8 * math.pi / 9, 10 * math.pi / 9),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    left_arm_action: ActionTerm = MISSING
    right_arm_action: ActionTerm = MISSING
    waist_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group.
        
        观测空间（共62维）：
        - left_joint_pos: 7维（左臂关节位置）
        - right_joint_pos: 7维（右臂关节位置）
        - waist_joint_pos: 3维（腰部关节位置）
        - left_joint_vel: 7维（左臂关节速度）
        - right_joint_vel: 7维（右臂关节速度）
        - waist_joint_vel: 3维（腰部关节速度）
        - left_pose_command: 7维（左臂目标位姿）
        - right_pose_command: 7维（右臂目标位姿）
        - left_actions: 7维（左臂上一步动作）
        - right_actions: 7维（右臂上一步动作）
        """

        # 左臂关节位置 (7 DOF)
        left_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint", "left_elbow_joint",
                        "left_wrist_roll_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint",
                    ],
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        # 右臂关节位置 (7 DOF)
        right_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                        "right_shoulder_yaw_joint", "right_elbow_joint",
                        "right_wrist_roll_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint",
                    ],
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        # 腰部关节位置 (3 DOF)
        waist_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        # 左臂关节速度 (7 DOF)
        left_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint", "left_elbow_joint",
                        "left_wrist_roll_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint",
                    ],
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        # 右臂关节速度 (7 DOF)
        right_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                        "right_shoulder_yaw_joint", "right_elbow_joint",
                        "right_wrist_roll_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint",
                    ],
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        # 腰部关节速度 (3 DOF)
        waist_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        # 目标位姿命令
        left_pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "left_ee_pose"})
        right_pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "right_ee_pose"})

        # 上一步动作
        left_actions = ObsTerm(func=mdp.last_action, params={"action_name": "left_arm_action"})
        right_actions = ObsTerm(func=mdp.last_action, params={"action_name": "right_arm_action"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.
    
    奖励设计：
    - 末端位置追踪：主要奖励，驱动手臂到达目标
    - 末端姿态追踪：次要奖励，保持正确姿态
    - 腰部惩罚：较大权重，鼓励优先使用手臂
    - 动作平滑性：避免剧烈变化
    """

    # 末端位置追踪
    left_end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "left_ee_pose",
        },
    )

    right_end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "right_ee_pose",
        },
    )

    left_end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "std": 0.1,
            "command_name": "left_ee_pose",
        },
    )

    right_end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "std": 0.1,
            "command_name": "right_ee_pose",
        },
    )

    # 末端姿态追踪
    left_end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "left_ee_pose",
        },
    )

    right_end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "right_ee_pose",
        },
    )

    # 动作平滑性惩罚
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)

    # 手臂关节速度惩罚
    left_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint", "left_elbow_joint",
                    "left_wrist_roll_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint",
                ],
            )
        },
    )
    right_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint", "right_elbow_joint",
                    "right_wrist_roll_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint",
                ],
            )
        },
    )

    # 腰部运动惩罚（较大权重，鼓励优先使用手臂）
    waist_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
            )
        },
    )

    # 腰部位置惩罚（鼓励腰部保持在零位附近）
    waist_joint_pos = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.0005,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
            )
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500},
    )

    left_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "left_joint_vel", "weight": -0.001, "num_steps": 4500},
    )

    right_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "right_joint_vel", "weight": -0.001, "num_steps": 4500},
    )

    waist_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "waist_joint_vel", "weight": -0.005, "num_steps": 4500},
    )


##
# Environment configuration
##


@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 24.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0
