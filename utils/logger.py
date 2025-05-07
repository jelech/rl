from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import os

from ..config import TrainingConfig


class BaseLogger(ABC):
    def __init__(self, cfg: TrainingConfig):
        self.step = 0
        self.cfg = cfg

    @abstractmethod
    def log(self, metrics: Dict[str, float]) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    def increment_step(self) -> None:
        self.step += 1


class TensorboardLogger(BaseLogger):
    def __init__(self, cfg: TrainingConfig):
        super().__init__(cfg)
        from torch.utils.tensorboard import SummaryWriter  # type: ignore

        os.makedirs(cfg.log_dir, exist_ok=True)
        self.writer = SummaryWriter(cfg.log_dir)

    def log(self, metrics: Dict[str, float]) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, self.step)
        self.increment_step()

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()


class MLflowLogger(BaseLogger):
    def __init__(self, cfg: TrainingConfig):
        super().__init__(cfg)
        import mlflow  # type: ignore

        self.mlflow = mlflow
        mlflow.set_experiment(cfg.mlflow_experiment)
        self.run = mlflow.start_run(run_name=cfg.run_name or "simple_rl")

    def log(self, metrics: Dict[str, float]) -> None:
        for key, value in metrics.items():
            self.mlflow.log_metric(key, value, step=self.step)
        self.increment_step()

    def close(self) -> None:
        self.mlflow.end_run()


class WandbLogger(BaseLogger):
    def __init__(self, cfg: TrainingConfig):
        super().__init__(cfg)
        try:
            import wandb  # type: ignore

            self.wandb = wandb

            # 初始化wandb
            wandb.init(project=cfg.wandb_project or "rl_training", name=cfg.run_name or None, config=vars(cfg))
        except ImportError:
            raise ImportError("please wandb: pip install wandb")

    def log(self, metrics: Dict[str, float]) -> None:
        metrics["step"] = self.step
        self.wandb.log(metrics)
        self.increment_step()

    def close(self) -> None:
        self.wandb.finish()


class NullLogger(BaseLogger):
    def log(self, metrics: Dict[str, float]) -> None:
        self.increment_step()

    def close(self) -> None:
        pass


class LoggerFactory:

    @staticmethod
    def create(cfg: TrainingConfig, active: bool = True) -> BaseLogger:
        if not active:
            return NullLogger(cfg)

        backend = cfg.log_backend.lower()

        if backend == "tensorboard":
            return TensorboardLogger(cfg)
        elif backend == "mlflow":
            return MLflowLogger(cfg)
        elif backend == "wandb":
            return WandbLogger(cfg)
        else:
            raise ValueError(f"Unknown backend: {backend}")


class Logger:
    def __init__(self, cfg: TrainingConfig, active: bool = True):
        self.logger = LoggerFactory.create(cfg, active)
        self.step = 0

    def log(self, metrics: Dict[str, float]) -> None:
        self.logger.log(metrics)
        self.step = self.logger.step

    def close(self) -> None:
        self.logger.close()
