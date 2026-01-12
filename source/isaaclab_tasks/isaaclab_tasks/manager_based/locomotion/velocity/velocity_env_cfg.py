# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# 预设配置
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# 场景定义
##

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """带有拟人化机器人和地形的交互场景配置。

    该类继承自 InteractiveSceneCfg，表示这是一个可交互的仿真场景。
    主要包含地形、机器人、传感器与光照等场景级别的配置项。
    """

    # 地面地形配置，使用 TerrainImporterCfg 来导入或生成地形
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",  # 地形在 USD 场景中的路径（prim 路径）
        terrain_type="generator",  # 使用生成器生成地形（而非导入静态高度图）
        terrain_generator=ROUGH_TERRAINS_CFG,  # 选择预定义的“粗糙地形”生成器配置
        max_init_terrain_level=5,  # 初始化时允许的最大地形难度等级（课程学习时有用）
        collision_group=-1,  # 碰撞分组（-1 表示默认或不分组）
        physics_material=sim_utils.RigidBodyMaterialCfg(
            # 刚体材料参数（用于物理仿真碰撞响应）
            friction_combine_mode="multiply",  # 摩擦系数合并方式
            restitution_combine_mode="multiply",  # 恢复系数合并方式
            static_friction=1.0,  # 静摩擦系数
            dynamic_friction=1.0,  # 动摩擦系数
        ),
        visual_material=sim_utils.MdlFileCfg(
            # 可视化材质，使用 Isaac Nucleus 中的 MDL 文件做纹理贴图
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,  # 启用 UV 投影以适配地形网格
            texture_scale=(0.25, 0.25),  # 纹理缩放因子（U, V）
        ),
        debug_vis=False,  # 是否开启地形调试可视化（例如高度图等）
    )
    # 机器人配置，目前在上层调用时必须提供该配置（MISSING 表示必填）
    robot: ArticulationCfg = MISSING

    # 高度扫描传感器配置（用于采集地形高度，供观察或导航使用）
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",  # 传感器挂载的 prim 路径模板
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),  # 传感器相对基座的偏移（向上 20m）
        ray_alignment="yaw",  # 光线的对齐方式（y轴旋转）
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),  # 扫描点格栅配置
        debug_vis=False,  # 是否显示扫描射线的调试信息
        mesh_prim_paths=["/World/ground"],  # 射线碰撞检测的目标 mesh（地形）
    )

    # 接触力传感器配置，监听机器人各部分（足端等）与环境接触信息
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # 说明：
    # - prim_path 支持通配符，用于匹配机器人下多个部位（如腿、脚）
    # - history_length: 保存接触历史的长度（用于计算接触时间、抖动等）
    # - track_air_time: 是否跟踪“离地时间”（用于奖励计算如 feet_air_time）

    # 天空光（环境光照）设置，使用 DomeLight 提供环境 HDRI 光照
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,  # 光强度（数值越大越亮）
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            # 使用 Nucleus 中的 HDR 天空贴图以获得逼真光照和环境反射
        ),
    )


##
# MDP 设置
##

@configclass
class CommandsCfg:
    """MDP 中命令（task command）的配置。

    在轨迹跟踪任务中，command 通常会指定目标速度（线速度、角速度）或方向等。
    这里使用 mdp.UniformVelocityCommandCfg 来随机/均匀采样目标速度。
    """

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",  # 命令对应的资产（机器人）名字，用于在环境中找对应实体
        resampling_time_range=(2.0, 10.0),  # 每隔多少秒重采样一次命令（固定为 10s）
        rel_standing_envs=0.1,  # 与“静止”相关的环境比例（可能用于命令采样策略）
        rel_heading_envs=1.0,  # 是否包含朝向命令的环境比例
        heading_command=True,  # 启用朝向（heading）命令
        heading_control_stiffness=0.5,  # 朝向控制器的刚性系数（用于软约束方向）
        debug_vis=True,  # 是否可视化命令（如朝向向量）
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            # 定义线速度和角速度的采样范围
            lin_vel_x=(-1.0, 1.0),  # 前后速度范围（m/s）
            lin_vel_y=(-1.0, 1.0),  # 侧向速度范围（m/s）
            ang_vel_z=(-1.0, 1.0),  # 角速度范围（rad/s）
            heading=(-math.pi, math.pi),  # 朝向角范围（rad）
        ),
    )


@configclass
class ActionsCfg:
    """定义智能体可执行动作的配置。

    这里把动作空间设置为关节位置控制（joint position），并指定缩放与默认偏置行为。
    """

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)
    # 说明：
    # - asset_name: 关联的机器人资产
    # - joint_names: 正则表达式匹配想要控制的关节，".*" 表示全部关节
    # - scale: 动作缩放系数，通常将网络输出缩放到物理可接受范围
    # - use_default_offset: 是否以关节默认位置作为偏移（常用于位置控制以避免零位不合理）


@configclass
class PolicyCfg(ObsGroup):
    """策略所需的观测组配置（仅使用真实传感器可获取的观测）。

    ObsGroup 表示一组有序的观测项，框架会按定义顺序合并（concatenate）或独立输出。
    
    真实传感器可获取的观测：
    - IMU: base_ang_vel（角速度）, projected_gravity（投影重力/姿态）, base_lin_acc（线加速度）
    - 关节编码器: joint_pos（关节位置）, joint_vel（关节速度）
    - 命令: velocity_commands（速度命令）
    - 历史动作: actions（上一步动作）
    """

    # IMU 观测
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        noise=Unoise(n_min=-0.2, n_max=0.2),
    )
    # 角速度观测：底座绕自身轴的角速度（带噪声）- 来自 IMU 陀螺仪

    base_lin_acc = ObsTerm(
        func=mdp.base_lin_acc,
        noise=Unoise(n_min=-0.1, n_max=0.1),
    )
    # 线加速度观测：底座在自身坐标系下的线加速度（带噪声）- 来自 IMU 加速度计

    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.1, n_max=0.1),
    )
    # 投影重力（机器人姿态）- 来自 IMU 加速度计

    # 命令观测
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    # 观测当前下发的速度命令（从 CommandsCfg 中生成）

    # 关节编码器观测
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.05, n_max=0.05))
    # 相对于参考位置的关节角度观测（带小幅噪声）- 来自关节编码器

    joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
    # 关节速度观测（带噪声）- 来自关节编码器微分或直接测量

    # 历史动作
    actions = ObsTerm(func=mdp.last_action)
    # 观测上一步的动作，常用于训练带时间相关性的策略

    def __post_init__(self):
        # 启用观测扰动选项（如果提供噪声则作用）
        self.enable_corruption = True
        # 将所有观测拼接成单个向量（concatenate_terms = True 表示合并为一维向量）
        self.concatenate_terms = True


@configclass
class ObservationsCfg:
    """观察项（obs）的配置集合。

    包含一个名为 PolicyCfg 的观测组（Observation Group），该组定义了策略所需的所有观测项。
    仅使用真实传感器可获取的观测（IMU + 关节编码器）。
    """

    @configclass
    class PolicyCfgWithHeightScan(PolicyCfg):
        """带高度扫描的策略观测组配置（用于粗糙地形）。"""

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        # 地形高度扫描观测：
        # - 通过 SceneEntityCfg 引用场景中定义的 height_scanner 传感器
        # - 对扫描值添加均匀噪声并限制输出范围到 [-1, 1]

    # 实际使用的观测组实例（可以在上层环境配置中替换或修改）
    policy: PolicyCfgWithHeightScan = PolicyCfgWithHeightScan()


@configclass
class EventCfg:
    """事件系统配置，包含启动、重置、间隔触发的事件。

    EventTerm 用于在仿真流程的不同阶段（startup, reset, interval）执行指定函数以实现随机化、
    干扰或者状态初始化。mdp 模块提供一系列通用函数，方便对机器人和环境参数进行改变。
    """

    # 启动时执行的事件（用于随机化刚体材料参数）
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",  # 在仿真启动（或场景加载）时执行
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),  # 对机器人所有刚体进行随机化
            "static_friction_range": (0.8, 0.8),  # 静摩擦范围（此处固定为 0.8）
            "dynamic_friction_range": (0.6, 0.6),  # 动摩擦范围（固定）
            "restitution_range": (0.0, 0.0),  # 恢复系数范围（固定）
            "num_buckets": 64,  # 随机化时使用的桶数量，用于离散化随机化值
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),  # 仅对 base 链接应用质量变化
            "mass_distribution_params": (-1.0, 3.0),  # 质量变化范围（相对或绝对取决于实现）
            "operation": "add",  # 操作类型（例如 add 表示增加质量）
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05)},
            # 质心（center of mass）偏移范围（米级），用于增加动力学多样性
        },
    )

    # 重置时执行的事件（用于每次环境 reset 时设置初始状态）
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),  # 固定为 0 表示不施加随机外力
            "torque_range": (-0.0, 0.0),  # 固定为 0（无随机扭矩）
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.02, 0.08),
                "roll": (-0.15, 0.15),
                "pitch": (-0.15, 0.15),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),  # 以比例或比例缩放的方式设置关节初始位置
            "velocity_range": (0.0, 0.0),  # 关节初始速度（此处固定为 0）
        },
    )

    # 间隔触发事件（用于给予周期性的扰动，比如推力）
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",  # 在 interval 模式下周期性触发
        interval_range_s=(0.0, 5.0),  # 触发时间间隔范围（随机或固定采样）
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},  # 推力对应的速度范围
    )

    # 惯量随机化
    inertia_randomization = EventTerm(
        func=mdp.randomize_inertia_properties,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "inertia_distribution_params": (0.5, 2.0),
            "armature_distribution_params": (0.5, 2.0),
            "operation": "scale",
        },
    )


@configclass
class RewardsCfg:
    """奖励与惩罚项配置。

    使用一系列 RewTerm 来描述任务奖励（鼓励向目标速度移动）与惩罚（抑制不良行为）。
    weight 的正负决定该项是奖励还是惩罚。
    """

    # 任务奖励项：鼓励跟踪线速度（xy 平面）
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # 任务奖励项：鼓励跟踪角速度（z 轴）
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # 惩罚项，降低不期望的行为
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)  # 垂直速度惩罚，抑制上下跳动
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)  # 侧向角速度惩罚，抑制滚转/俯仰角速
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)  # 关节力矩大小惩罚（节能）
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)  # 关节加速度惩罚（平滑动作）
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)  # 动作变化率惩罚（防止突变）
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    # feet_air_time: 鼓励在符合命令的前提下足端有适当的离地时间（用于步态质量衡量）
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )
    # undesired_contacts: 对不期望部位（如大腿）与地面接触的惩罚

    # 可选额外惩罚（当前权重设为 0，表示禁用，但保留配置便于后续调整）
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)  # 鼓励机身保持平坦
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)  # 关节位置限制惩罚


@configclass
class TerminationsCfg:
    """终止条件配置。

    DoneTerm 用于定义导致 episode 终止的条件，例如时间到或非法接触等。
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)  # 基于时间的终止（由环境 episode_length_s 控制）
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    # base_contact: 如果机器人底座与地面产生非法接触（例如碰撞），则终止 episode


@configclass
class CurriculumCfg:
    """课程学习相关配置项。

    课程（curriculum）用于逐步增加任务难度，例如增加地形复杂度或速度目标范围。
    """

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    # terrain_levels: 通过 mdp.terrain_levels_vel 定义如何根据训练进度切换地形难度


##
# 环境总体配置
##

@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """轨迹跟踪的粗糙地形上运动环境配置。

    该配置组合了场景、观测、动作、奖励、事件、终止条件与课程学习的所有子配置，
    并在 __post_init__ 中完成一些与仿真时间步、传感器更新频率等相关的联动设置。
    """

    # 场景设置：创建 MySceneCfg 实例并设置并行环境数量与环境间距
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # 观察、动作、命令配置
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP 相关组件（奖励、终止、事件、课程）
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """在完成初始化之后自动运行的配置补充逻辑。

        这里用于设置仿真基础参数（decimation、时间步长）、传感器更新频率、
        物理材质关联以及根据是否启用了课程调整地形生成器行为。
        """
        # 通用设定
        self.decimation = 4  # 控制渲染或控制更新的下采样（动作/策略每 decimation 步应用一次）
        self.episode_length_s = 20.0  # 每个 episode 的时长（秒）

        # 仿真设置：时间步长与渲染间隔
        self.sim.dt = 0.005  # 物理仿真时间步长（秒）
        self.sim.render_interval = self.decimation  # 渲染/策略更新间隔（以 step 计）
        # 将场景地形的物理材料应用到全局仿真设置（便于统一摩擦等参数）
        self.sim.physics_material = self.scene.terrain.physics_material
        # 提高 PhysX 中 GPU 刚体补丁的最大数量（与并行环境数量、地形复杂度相关）
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # 升级传感器更新周期（以物理时钟为准）
        if self.scene.height_scanner is not None:
            # 高度扫描器的更新周期设为 decimation * sim.dt（与策略决策频率对齐）
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            # 接触力传感器更新更频繁，设为每个物理步更新
            self.scene.contact_forces.update_period = self.sim.dt

        # 如果地形难度课程被启用，则提前告知地形生成器（terrain_generator）使用课程功能
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                # 启用地形生成器的课程模式，以便在训练过程中动态调整难度
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                # 否则关闭课程模式，使用固定难度
                self.scene.terrain.terrain_generator.curriculum = False
