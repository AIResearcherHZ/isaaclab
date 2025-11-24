from isaaclab.managers import SceneEntityCfg
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