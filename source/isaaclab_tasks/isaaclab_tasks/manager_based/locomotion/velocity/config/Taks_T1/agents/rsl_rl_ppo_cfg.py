from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class TaksT1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Taks_T1 粗糙地形 PPO 训练配置。
    
    仅使用真实传感器可获取的观测：
    - IMU: base_ang_vel, projected_gravity
    - 关节: joint_pos, joint_vel
    """
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "Taks_T1_rough"
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"],
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class TaksT1FlatPPORunnerCfg(TaksT1RoughPPORunnerCfg):
    """Taks_T1 平坦地形 PPO 训练配置。"""

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 2500
        self.experiment_name = "Taks_T1_flat"
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]
