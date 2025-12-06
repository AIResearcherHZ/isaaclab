from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    EventCfg,
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

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
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )

    # 颈部关节偏差惩罚 - 保持头部稳定
    joint_deviation_neck = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["neck_.*"])},
    )

    # 躯干pitch惩罚 - 防止过度前倾或后仰
    body_pitch_penalty = RewTerm(
        func=mdp.body_pitch_penalty,
        weight=-0.1,
        params={"max_pitch": 0.30},
    )

    # 腰部偏差惩罚：抑制躯干晃动，保持腰部姿态稳定
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_pitch_joint", "waist_yaw_joint", "waist_roll_joint"])},
    )

    # 手臂关节偏差惩罚：减少上肢多余摆动，保持动作干净
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_.*"],
            )
        },
    )

    # 手臂俯仰轴扭矩惩罚：限制肩部 pitch 轴扭矩，避免动作过猛
    arm_torque_penalty_pitch = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_pitch_joint"])},
    )

    # 其余手臂关节扭矩惩罚：使非 pitch 轴保持低扭矩，避免抖动
    arm_torque_penalty_others = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.5e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_.*"])}
    )
    
    # 腰部扭矩惩罚：限制腰部扭矩，避免动作过猛
    waist_torques_penalty_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-5e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_pitch_joint", "waist_yaw_joint", "waist_roll_joint"])},
    )

    # 颈部扭矩惩罚：限制颈部扭矩，避免动作过猛
    neck_torques_penalty_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-5e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["neck_pitch_joint", "neck_yaw_joint", "neck_roll_joint"])},
    )

    # 步态对称性奖励 - 鼓励左右脚交替接触
    gait_symmetry = RewTerm(
        func=mdp.gait_symmetry,
        weight=0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )

    # 双脚同时接触惩罚 - 防止双脚同时离地或同时着地过久
    double_support_penalty = RewTerm(
        func=mdp.double_support_time_penalty,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "max_double_support_time": 0.2,
        }
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
        weight=0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "command_name": "base_velocity",
        },
    )

    # 双脚交替接触奖励 - 鼓励一脚着地一脚离地
    feet_alternating = RewTerm(
        func=mdp.feet_alternating_contact,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "command_name": "base_velocity",
        },
    )

    # 静止姿态奖励 - 无命令时保持标准站姿
    stand_still_posture = RewTerm(
        func=mdp.stand_still_posture,
        weight=0.5,
        params={"command_name": "base_velocity", "command_threshold": 0.1},
    )

    # 静止时关节偏差惩罚 - 当命令接近零时保持关节在默认位置
    stand_still_joint_deviation = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-0.25,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 方向切换惩罚 - 当从前进变后退时惩罚过快变化
    direction_change_penalty = RewTerm(
        func=mdp.command_direction_change_penalty,
        weight=-0.1,
        params={"command_name": "base_velocity"},
    )

    # 速度方向对齐奖励 - 鼓励实际速度与命令方向一致
    velocity_alignment = RewTerm(
        func=mdp.velocity_direction_alignment,
        weight=0.02,
        params={"command_name": "base_velocity"},
    )

@configclass
class TaksT1EventCfg(EventCfg):
    """域随机化配置，包含电机老化、关节摩擦等corner case。"""

    # 关节摩擦随机化 - 模拟关节磨损
    randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "armature_distribution_params": (0.1, 2.0),  # 电机转子惯量缩放
            "operation": "scale",
        },
    )

    # 刚体惯性随机化 - 模拟连杆惯性不确定性
    randomize_body_inertia = EventTerm(
        func=mdp.randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "inertia_distribution_params": (0.1, 2.0),  # 刚体惯性缩放
            "operation": "scale",
        },
    )

    # ==================== 新增鲁棒性随机化（极低频率 corner case） ====================

    # 动作噪声 - 模拟控制信号不完美（量化误差、通讯抖动）
    action_noise = EventTerm(
        func=mdp.randomize_action_noise,
        mode="interval",
        interval_range_s=(40.0, 60.0),  # 40-60秒触发一次
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "noise_std": 0.05,
            "noise_type": "gaussian",
        },
    )

    # 动作延迟 - 模拟通讯延迟和控制周期不对齐
    action_delay = EventTerm(
        func=mdp.randomize_action_delay,
        mode="interval",
        interval_range_s=(40.0, 60.0),  # 40-60秒触发一次
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_delay_steps": 5,  # 最大延迟5步
        },
    )

    # 关节编码器噪声 - 模拟编码器测量误差和零点偏移
    encoder_noise = EventTerm(
        func=mdp.randomize_joint_encoder_noise,
        mode="interval",
        interval_range_s=(40.0, 60.0),  # 40-60秒触发一次
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pos_noise_std": 0.01,  # 位置噪声标准差 (rad)
            "vel_noise_std": 0.02,   # 速度噪声标准差 (rad/s)
            "pos_bias_range": (-0.05, 0.05),  # 位置偏置范围 (rad)
            "vel_bias_range": (-0.1, 0.1),  # 速度偏置范围 (rad/s)
        },
    )

    # IMU噪声和漂移 - 模拟真实IMU的测量特性
    imu_noise = EventTerm(
        func=mdp.randomize_imu_noise_and_bias,
        mode="interval",
        interval_range_s=(40.0, 60.0),  # 40-60秒触发一次
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "ang_vel_noise_std": 0.05,  # 角速度噪声 (rad/s)
            "lin_acc_noise_std": 0.1,  # 线加速度噪声 (m/s^2)
            "ang_vel_bias_range": (-0.1, 0.1),
            "lin_acc_bias_range": (-0.2, 0.2),
            "bias_drift_std": 0.05,  # 偏置漂移
        },
    )

    # 观测丢包 - 模拟传感器偶发失效
    observation_dropout = EventTerm(
        func=mdp.randomize_observation_dropout,
        mode="interval",
        interval_range_s=(40.0, 60.0),  # 40-60秒触发一次
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "dropout_prob": 0.005,  # 每个维度丢包概率 0.5%
            "dropout_mode": "hold",  # 丢包时保持上一帧值
        },
    )

    # 关节故障 - 模拟电机故障（极低概率）
    joint_failure = EventTerm(
        func=mdp.randomize_joint_failure,
        mode="reset",  # 每次reset时重新采样故障状态
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "failure_prob": 0.0005,  # 每个关节失效概率 0.05%
            "failure_mode": "weak",  # 弱化模式（扭矩衰减）
            "weak_factor": 0.5,  # 衰减因子提高，故障程度减轻
        },
    )

    # 传感器延迟尖峰 - 模拟偶发的通讯阻塞
    sensor_latency_spike = EventTerm(
        func=mdp.randomize_sensor_latency_spike,
        mode="interval",
        interval_range_s=(40.0, 60.0),  # 40-60秒触发一次
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "spike_prob": 0.005,  # 0.5%概率发生延迟尖峰
            "max_latency_steps": 10,  # 最大延迟10步
        },
    )

    # 重力方向偏置 - 模拟基座倾斜/坡度
    slope_randomization = EventTerm(
        func=mdp.randomize_slope_or_base_frame,
        mode="startup",  # 仿真开始时设置
        params={
            "gravity_bias_range": {
                "x": (-0.1, 0.1),  # x方向重力偏置 (m/s^2)
                "y": (-0.1, 0.1),  # y方向重力偏置 (m/s^2)
                "z": (-0.05, 0.05),  # z方向重力偏置 (m/s^2)
            },
        },
    )

@configclass
class TaksT1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "torso_link|pelvis"
    foot_link_name = ".*_ankle_roll_link"

    rewards: TaksT1Rewards = TaksT1Rewards()
    # 使用扩展的事件配置（包含电机老化、关节摩擦等域随机化）
    events: TaksT1EventCfg = TaksT1EventCfg()

    def __post_init__(self):
        # 调用父类后初始化逻辑，确保基础配置正确设置
        super().__post_init__()

        # ------------------------------Scene------------------------------
        self.scene.robot = TAKS_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations------------------------------
        # 只使用真实传感器可获取的观测（IMU + 关节编码器）
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        # 删除无法真实获取的观测
        self.observations.policy.height_scan = None
        self.observations.policy.base_lin_vel = None

        # ------------------------------Actions------------------------------
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}

        # ------------------------------Events------------------------------
        self.events.push_robot.params["velocity_range"] = {"x": (-2.5, 2.5), "y": (-2.5, 2.5)}
        self.events.push_robot.interval_range_s = (0.0, 5.0)
        self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names=self.base_link_name)
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.base_external_force_torque.params["force_range"] = (-2.5, 2.5)
        self.events.base_external_force_torque.params["torque_range"] = (-2.0, 2.0)

        # 重置机器人关节时增加随机性
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (1.0, 1.0)
        
        # 重置底座时增加初始速度随机化
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.05, 0.15),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            },
        }
        self.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names=".*")
        self.events.base_com.params["com_range"] = {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)}

        # 机器人摩擦力随机化 - 只对脚踝关节应用
        self.events.physics_material.params["asset_cfg"] = SceneEntityCfg("robot", body_names=self.foot_link_name)
        self.events.physics_material.params["static_friction_range"] = (0.1, 2.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.1, 2.0)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.5)
        self.events.physics_material.params["num_buckets"] = 64

        # ------------------------------Rewards------------------------------
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.dof_acc_l2.weight = -5.0e-7
        self.rewards.dof_torques_l2.weight = -5.0e-5
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # ------------------------------Terminations------------------------------
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            self.base_link_name,
            ".*_shoulder_pitch_link",
            ".*_shoulder_roll_link",
            ".*_shoulder_yaw_link",
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

        # 关闭所有新增的鲁棒性随机化事件（调试用）
        self.events.action_noise = None
        self.events.action_delay = None
        self.events.encoder_noise = None
        self.events.imu_noise = None
        self.events.observation_dropout = None
        self.events.joint_failure = None
        self.events.sensor_latency_spike = None
        self.events.slope_randomization = None
        self.events.randomize_body_inertia = None

        # 启用场景查询支持,用于碰撞检测和射线投射等功能
        self.sim.enable_scene_query_support = True

