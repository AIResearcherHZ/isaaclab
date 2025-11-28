import gymnasium as gym

from . import agents

##
# 注册 Gym 环境。
##

# 注册粗略地形版本的 G1 机器人速度控制任务环境
gym.register(
    id="Isaac-Velocity-Rough-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # 环境配置入口：指向当前模块下的 rough_env_cfg 中的 G1RoughEnvCfg
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughEnvCfg",
        # RSL RL PPO 训练器配置：指向 agents 模块下对应的配置类
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        # SKRL 配置文件入口：指向 agents 模块下 skrl_rough_ppo_cfg.yaml 文件
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

# 注册粗略地形（试玩模式）版本的 G1 机器人速度控制任务环境
gym.register(
    id="Isaac-Velocity-Rough-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # 环境配置入口：试玩模式下使用 G1RoughEnvCfg_PLAY
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

# 注册粗糙地形 Teacher-Student 蒸馏环境
gym.register(
    id="Velocity-Rough-G1-Distillation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughTeacherStudentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1VelocityDistillationRunnerCfg",
    },
)

# 注册粗糙地形 Student 微调环境
gym.register(
    id="Velocity-Rough-G1-Student-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughStudentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatStudentPPORunnerCfg",
    },
)

# 注册平坦地形版本的 G1 机器人速度控制任务环境
gym.register(
    id="Isaac-Velocity-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # 环境配置入口：平坦地形主环境配置类
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg",
        # RSL PPO 配置使用平坦环境专用的 Runner
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        # SKRL 配置文件：平坦地形对应的 yaml
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

# 注册平坦地形（试玩模式）版本的 G1 机器人速度控制任务环境
gym.register(
    id="Isaac-Velocity-Flat-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # 环境配置入口：试玩模式下使用 G1FlatEnvCfg_PLAY
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)


###############################
# Teacher-Student Distillation #
###############################

# 注册 Teacher-Student 蒸馏环境
# 用于从 Teacher 策略（使用特权观测）蒸馏到 Student 策略（使用非特权观测）
gym.register(
    id="Velocity-G1-Distillation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatTeacherStudentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1VelocityDistillationRunnerCfg",
    },
)


########################
# Student Fine-tune #
########################

# 注册 Student 微调环境
# 用于使用 RL 进一步微调 Student 策略（仅使用真实传感器可获取的观测）
gym.register(
    id="Velocity-G1-Student-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatStudentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatStudentPPORunnerCfg",
    },
)