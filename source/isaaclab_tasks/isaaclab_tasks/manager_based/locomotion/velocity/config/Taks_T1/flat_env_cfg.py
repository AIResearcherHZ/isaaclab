from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    StudentObservationsCfg,
    TeacherStudentObservationsCfg,
)

from .rough_env_cfg import TaksT1RoughEnvCfg


@configclass
class TaksT1FlatEnvCfg(TaksT1RoughEnvCfg):
    def __post_init__(self):
        # 调用父类的 __post_init__ 以确保基类配置初始化完成
        super().__post_init__()

        # 将地形切换为平面
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # 不使用地面高度扫描器
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # 取消地形课程机制，因为地形固定为平坦
        self.curriculum.terrain_levels = None
        # 奖励函数配置部分
        self.rewards.feet_air_time.weight = 0.75
        # 命令空间限制配置：限制线速度和角速度范围
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class TaksT1FlatEnvCfg_PLAY(TaksT1FlatEnvCfg):
    def __post_init__(self) -> None:
        # 调用父类的后置初始化以确保基础配置有效
        super().__post_init__()

        # 为调试/试玩模式缩小场景规模
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # 试玩时关闭观测扰动，避免引入随机性
        self.observations.policy.enable_corruption = False
        # 移除试验中对机器人的随机推动事件
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

        # 启用场景查询支持,用于碰撞检测和射线投射等功能
        self.sim.enable_scene_query_support = True


###############################
# Teacher-Student Distillation #
###############################


@configclass
class TaksT1FlatTeacherStudentEnvCfg(TaksT1FlatEnvCfg):
    """Teacher-Student 蒸馏阶段的环境配置。

    该配置用于行为克隆（Behavior Cloning）阶段：
    - Student 策略从 Teacher 策略学习
    - Teacher 使用包含特权信息（如 base_lin_vel）的观测
    - Student 使用不含特权信息的观测
    """

    observations: TeacherStudentObservationsCfg = TeacherStudentObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # 蒸馏阶段使用较少的环境数量
        self.scene.num_envs = 256

        # 蒸馏阶段降低 Teacher 观测噪声，提高学习稳定性
        self.observations.teacher.base_lin_vel.noise = Unoise(n_min=-0.001, n_max=0.001)
        self.observations.teacher.base_ang_vel.noise = Unoise(n_min=-0.002, n_max=0.002)
        self.observations.teacher.projected_gravity.noise = Unoise(n_min=-0.0005, n_max=0.0005)
        self.observations.teacher.joint_pos.noise = Unoise(n_min=-0.0001, n_max=0.0001)
        self.observations.teacher.joint_vel.noise = Unoise(n_min=-0.0001, n_max=0.0001)


########################
# Student Fine-tune #
########################


@configclass
class TaksT1FlatStudentEnvCfg(TaksT1FlatEnvCfg):
    """Student 微调阶段的环境配置。

    该配置用于 Student 策略的 RL 微调阶段：
    - 仅使用真实传感器可获取的观测（不含 base_lin_vel）
    - 通过 RL 进一步优化 Student 策略性能
    """

    observations: StudentObservationsCfg = StudentObservationsCfg()