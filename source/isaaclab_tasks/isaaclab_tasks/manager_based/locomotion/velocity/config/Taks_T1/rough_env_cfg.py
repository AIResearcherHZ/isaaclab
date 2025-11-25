from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

##
# 预定义配置
##
from isaaclab_assets import TAKS_T1_CFG  # isort: skip


@configclass
class TaksT1Rewards(RewardsCfg):
    """定义用于 MDP 训练中的奖励项。"""

    # 终止惩罚
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # 追踪线速度奖励
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # 追踪角速度奖励
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # 抬脚时间奖励
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )

    # 脚滑动惩罚
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    
    # 踝关节位置限制惩罚：若末端执行器超出设定范围则给予负奖励
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )

    # 髋部关节偏差惩罚
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )

    # 腰部关节偏差惩罚
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint", "waist_roll_joint"])},
    )

    # 颈部关节偏差惩罚 - 保持头部稳定
    joint_deviation_neck = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["neck_.*"])},
    )

    # 步态对称性奖励
    gait_symmetry = RewTerm(
        func=mdp.gait_symmetry,
        weight=0.25,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )

    # 静止奖励
    stand_still = RewTerm(
        func=mdp.stand_still_when_zero_command,
        weight=0.2,
        params={"command_name": "base_velocity", "command_threshold": 0.05},
    )

    # 手臂摆动奖励 - 鼓励手臂自然摆动以平衡下肢惯量
    arm_swing = RewTerm(
        func=mdp.arm_swing_reward,
        weight=0.1,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_pitch_joint", ".*_elbow_joint"]),
        },
    )


@configclass
class TaksT1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    # 使用上一节定义的奖励配置
    rewards: TaksT1Rewards = TaksT1Rewards()

    def __post_init__(self):
        # 调用父类后初始化逻辑，确保基础配置正确设置
        super().__post_init__()

        # 场景相关设置
        self.scene.robot = TAKS_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 高度扫描器指向机器人的躯干，用于动态仿真时采集高度信息
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # 保留推人事件，增加扰动自稳定训练
        self.events.push_robot.params["velocity_range"] = {"x": (-0.8, 0.8), "y": (-0.8, 0.8)}
        self.events.push_robot.interval_range_s = (8.0, 12.0)

        # 增加基座质量随机化
        self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names="torso_link")
        self.events.add_base_mass.params["mass_distribution_params"] = (-8.0, 8.0)

        # 增加关节初始位置的随机化范围
        self.events.reset_robot_joints.params["position_range"] = (0.8, 1.2)

        # 在躯干施加随机外力和扭矩，增强扰动抗性
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.base_external_force_torque.params["force_range"] = (-5.0, 5.0)
        self.events.base_external_force_torque.params["torque_range"] = (-2.0, 2.0)

        # 重置底座时增加初始速度随机化
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        }

        # 保留质心随机化以增加动力学多样性
        self.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="torso_link")
        self.events.base_com.params["com_range"] = {"x": (-0.08, 0.08), "y": (-0.08, 0.08), "z": (-0.02, 0.02)}

        # 奖励权重进一步细调
        self.rewards.undesired_contacts = None
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )

        # 适度惩罚腿部扭矩,但不过度限制
        self.rewards.dof_torques_l2.weight = -1.0e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )

        # 命令空间线速度与角速度设置
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # 终止条件：躯干、双臂、髋关节接触地面即终止
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "torso_link",  # 躯干
            ".*_shoulder_pitch_link",  # 肩部pitch关节
            ".*_shoulder_roll_link",  # 肩部roll关节
            ".*_shoulder_yaw_link",  # 肩部yaw关节
            ".*_elbow_link",  # 肘部
            ".*_wrist_roll_link",  # 腕部roll
            ".*_wrist_yaw_link",  # 腕部yaw
            ".*_wrist_pitch_link",  # 腕部pitch
            ".*_hip_pitch_link",  # 髋部pitch
        ]


@configclass
class TaksT1RoughEnvCfg_PLAY(TaksT1RoughEnvCfg):
    def __post_init__(self):
        # 调用父类后初始化
        super().__post_init__()

        # 试玩模式下减小环境数量与间距，便于观测
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # 玩耍模式保持较短的单集长度
        self.episode_length_s = 40.0
        # 不再通过地形等级初始化机器人，而是在网格上随机生成
        self.scene.terrain.max_init_terrain_level = None
        # 若存在地形生成器，减小地形格数并关闭课程
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # 强制机器人始终向前移动，不产生横向速度变动
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # 试玩模式关闭观测扰动，避免不确定性来源
        self.observations.policy.enable_corruption = False
        # 移除所有随机推力事件以便于调试
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # 启用场景查询支持,用于碰撞检测和射线投射等功能
        self.sim.enable_scene_query_support = True