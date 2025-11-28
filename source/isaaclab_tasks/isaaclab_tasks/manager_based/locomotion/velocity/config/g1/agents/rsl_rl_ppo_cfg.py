from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationStudentTeacherRecurrentCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class G1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 每个环境每个 rollout 收集的时间步数量（采样长度）
    num_steps_per_env = 24
    # 最大训练轮数
    max_iterations = 3000
    # 模型保存间隔（每 n 次迭代保存一次）
    save_interval = 50
    # 实验名称，用于日志/检查点组织
    experiment_name = "g1_rough"
    # 观测组配置：定义 policy 和 critic 使用的观测集合
    obs_groups = {
        "policy": ["policy"],  # policy 使用 "policy" 观测组
        "critic": ["policy"],  # critic 也使用 "policy" 观测组
    }
    # 策略网络配置：使用 PPO 的 Actor-Critic 结构
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,  # 随机初始化动作扰动标准差
        actor_obs_normalization=False,  # 不对 Actor 的观测做标准化
        critic_obs_normalization=False,  # 不对 Critic 的观测做标准化
        actor_hidden_dims=[512, 256, 128],  # Actor 网络各层宽度
        critic_hidden_dims=[512, 256, 128],  # Critic 网络各层宽度
        activation="elu",  # 全网络激活函数类型
    )
    # PPO 算法相关超参
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,  # Critic loss 的权重系数
        use_clipped_value_loss=True,  # 使用裁剪后的 value loss
        clip_param=0.2,  # PPO 裁剪阈值（epsilon）
        entropy_coef=0.008,  # 熵正则项系数，鼓励策略探索
        num_learning_epochs=5,  # 每个更新周期的学习轮数
        num_mini_batches=4,  # mini-batch 数量，用于将 rollout 拆分
        learning_rate=1.0e-3,  # 初始学习率
        schedule="adaptive",  # 学习率调度方式（adaptive 表示自适应调整）
        gamma=0.99,  # 折扣因子
        lam=0.95,  # GAE 的 λ 参数
        desired_kl=0.01,  # 期望的 KL 散度，用于自适应学习率调整
        max_grad_norm=1.0,  # 梯度裁剪阈值，避免梯度爆炸
    )


@configclass
class G1FlatPPORunnerCfg(G1RoughPPORunnerCfg):
    def __post_init__(self):
        # 调用父类的后初始化以确保所有配置被正确继承与处理
        super().__post_init__()

        # 平坦地形版本训练迭代次数减半，因任务简单
        self.max_iterations = 1500
        # 实验名称更新，用于区分不同训练场景的日志/模型
        self.experiment_name = "g1_flat"
        # 平坦环境下使用更小的网络规模以降低计算开销
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]


###############################
# Teacher-Student Distillation #
###############################


@configclass
class G1VelocityDistillationRunnerCfg(G1FlatPPORunnerCfg):
    """Teacher-Student 蒸馏阶段的训练配置。

    使用行为克隆（Behavior Cloning）从 Teacher 策略蒸馏到 Student 策略：
    - Teacher: 使用特权观测（包含 base_lin_vel）的 MLP 策略
    - Student: 使用非特权观测的 LSTM 循环策略
    """

    seed = 42
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    run_name = "distillation"

    # 蒸馏算法配置
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        gradient_length=5,
        learning_rate=1e-3,
        loss_type="mse",  # 使用 MSE 损失进行行为克隆
    )

    # Student-Teacher 网络配置
    policy = RslRlDistillationStudentTeacherRecurrentCfg(
        student_hidden_dims=[256, 256, 128],  # Student MLP 隐藏层
        teacher_hidden_dims=[256, 128, 128],  # Teacher MLP 隐藏层
        activation="elu",
        init_noise_std=0.1,
        class_name="StudentTeacherRecurrent",
        rnn_type="lstm",  # Student 使用 LSTM 处理时序信息
        rnn_hidden_dim=256,
        rnn_num_layers=3,
        teacher_recurrent=False,  # Teacher 不使用循环网络
    )

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1500
        self.experiment_name = "g1_distillation"


########################
# Student Fine-tune #
########################


@configclass
class G1FlatStudentPPORunnerCfg(G1FlatPPORunnerCfg):
    """Student 策略 RL 微调阶段的训练配置。

    在蒸馏完成后，使用 RL 进一步微调 Student 策略：
    - 仅使用真实传感器可获取的观测
    - 使用 LSTM 循环网络处理时序信息以补偿缺失的速度观测
    """

    # 使用循环 Actor-Critic 网络
    policy = RslRlPpoActorCriticRecurrentCfg(
        class_name="ActorCriticRecurrent",
        init_noise_std=0.1,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=2,
    )

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 4000
        self.run_name = "student_finetune"
        self.experiment_name = "g1_student_finetune"