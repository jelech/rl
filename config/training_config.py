from dataclasses import dataclass
from .base_config import BaseConfig


@dataclass
class AlgorithmConfig(BaseConfig):
    """算法配置"""

    algorithm: str = "ppo"  # "a2c", "ppo", "dqn"
    gamma: float = 0.99
    lr: float = 2.5e-4
    batch_size: int = 32
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5

    # PPO特有参数
    ppo_epochs: int = 4
    ppo_clip: float = 0.2


@dataclass
class ModelConfig(BaseConfig):
    """模型配置"""

    hidden_dim: int = 128
    activation: str = "relu"  # "relu", "tanh", "leaky_relu"
    shared_backbone: bool = True
    normalize_obs: bool = False


@dataclass
class TrainingConfig(BaseConfig):
    """训练配置"""

    seed: int = 42
    total_episodes: int = 300
    num_actors: int = 1

    # 分布式训练
    distributed: bool = False
    num_workers: int = 4
    num_envs_per_worker: int = 1
    use_gpu: bool = False
    ray: bool = False
    ddp: bool = False

    # 日志与评估
    log_backend: str = "tensorboard"  # "tensorboard", "mlflow", "wandb"
    log_dir: str = "runs/rl_trainer"
    mlflow_experiment: str = "rl_trainer"
    wandb_project: str = "rl_trainer"
    eval_interval: int = 20
    eval_episodes: int = 5
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 50
    run_name: str = "simple_rl"
