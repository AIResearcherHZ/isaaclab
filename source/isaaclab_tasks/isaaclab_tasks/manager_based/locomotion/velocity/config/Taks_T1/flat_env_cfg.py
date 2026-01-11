from isaaclab.utils import configclass

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
        self.rewards.feet_air_time.weight = 1.25
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
        # 试玩模式关闭观测扰动，避免不确定性来源
        self.observations.policy.enable_corruption = False
        # 移除所有随机推力事件以便于调试
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.feet_external_force_torque = None

        # 关闭所有新增的鲁棒性随机化事件（调试用）
        self.events.action_delay = None
        self.events.joint_failure = None
        self.events.inertia_randomization = None

        # 启用场景查询支持,用于碰撞检测和射线投射等功能
        self.sim.enable_scene_query_support = True
