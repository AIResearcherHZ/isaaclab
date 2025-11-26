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
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    
    # 踝关节位置限制惩罚：若末端执行器超出设定范围则给予负奖励
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )

    # 髋部关节偏差惩罚
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )

    # 颈部关节偏差惩罚 - 保持头部稳定
    joint_deviation_neck = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["neck_.*"])},
    )

    # 腰部关节偏差惩罚：保持躯干稳定，减少由腰部引起的晃动
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_pitch_joint", "waist_yaw_joint", "waist_roll_joint"])},
    )
    
    # 手臂关节偏差惩罚：减少上肢不必要摆动，保持干净的动作
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_roll_joint",
                ],
            )
        },
    )
    
    # 步态对称性奖励 - 鼓励左右脚交替接触
    gait_symmetry = RewTerm(
        func=mdp.gait_symmetry,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    
    # 双脚同时接触惩罚 - 防止双脚同时离地或同时着地过久
    double_support_penalty = RewTerm(
        func=mdp.double_support_time_penalty,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "max_double_support_time": 0.4,
        },
    )
    
    # 膝关节过度弯曲惩罚 - 防止蹲姿
    knee_bend_penalty = RewTerm(
        func=mdp.knee_bend_penalty,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_knee_joint"),
            "max_bend_angle": 0.78,
        },
    )

    # 单脚支撑奖励 - 鼓励正常迈步
    single_leg_stance = RewTerm(
        func=mdp.single_leg_stance_reward,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "command_name": "base_velocity",
        },
    )

    # 双脚交替接触奖励 - 鼓励一脚着地一脚离地
    feet_alternating = RewTerm(
        func=mdp.feet_alternating_contact,
        weight=0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "command_name": "base_velocity",
        },
    )

    # 静止时关节偏差惩罚 - 当命令接近零时保持关节在默认位置
    stand_still_joint_deviation = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-0.2,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.06,
            "asset_cfg": SceneEntityCfg("robot"),
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
        self.events.push_robot.params["velocity_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
        self.events.push_robot.interval_range_s = (0.0, 5.0)

        # 增加基座质量随机化
        self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names="torso_link")
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)

        # 在躯干施加随机外力和扭矩，增强扰动抗性
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.base_external_force_torque.params["force_range"] = (-2.5, 2.5)
        self.events.base_external_force_torque.params["torque_range"] = (-1.0, 1.0)

        # 重置机器人关节时增加随机性
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (1.0, 1.0)
        
        # 重置底座时增加初始速度随机化
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
        }

        # 保留质心随机化以增加动力学多样性
        self.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="torso_link")
        self.events.base_com.params["com_range"] = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.02, 0.02)}

        # ========== 地面摩擦力域随机化 ==========
        # 机器人脚部摩擦力随机化 (摩擦力必须 >= 0)
        self.events.physics_material.params["asset_cfg"] = SceneEntityCfg("robot", body_names=".*_ankle_roll_link")
        self.events.physics_material.params["static_friction_range"] = (0.6, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.4, 1.0)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.1)
        self.events.physics_material.params["num_buckets"] = 64

        # 奖励权重进一步细调
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.025
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*", "waist_.*", ".*_shoulder_.*"]
        )

        # 启用扭矩惩罚以减少振荡
        self.rewards.dof_torques_l2.weight = -1.25e-5

        # 命令空间线速度与角速度设置
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # 终止条件：躯干、双臂、髋关节接触地面即终止
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "pelvis",  # 骨盆
            "torso_link",  # 躯干
            ".*_shoulder_pitch_link",  # 肩部pitch关节
            ".*_shoulder_roll_link",  # 肩部roll关节
            ".*_shoulder_yaw_link",  # 肩部yaw关节
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

        # 命令空间线速度，角速度与高度设置
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # 试玩模式关闭观测扰动，避免不确定性来源
        self.observations.policy.enable_corruption = False
        # 移除所有随机推力事件以便于调试
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # 启用场景查询支持,用于碰撞检测和射线投射等功能
        self.sim.enable_scene_query_support = True