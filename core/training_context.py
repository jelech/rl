import torch
import random
from abc import ABC, abstractmethod

import torch.distributed as dist

from ..config import TrainingConfig


class TrainingContext(ABC):
    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self._setup()

    @abstractmethod
    def _setup(self):
        pass

    @abstractmethod
    def dist_wait(self):
        pass

    @property
    @abstractmethod
    def is_master(self) -> bool:
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass

    @abstractmethod
    def cleanup(self):
        pass


class SingleTrainingContext(TrainingContext):
    def __init__(self, cfg: TrainingConfig):
        super().__init__(cfg)

    def _setup(self):
        if torch.cuda.is_available():
            self._device = torch.device("cuda:0")
        else:
            self._device = torch.device("cpu")

    def dist_wait(self):
        pass

    @property
    def is_master(self) -> bool:
        return True

    @property
    def device(self) -> torch.device:
        return self._device

    def cleanup(self):
        pass


class DistributedTrainingContext(TrainingContext):
    def __init__(self, cfg: TrainingConfig):
        super().__init__(cfg)

    def _setup(self):
        assert dist is not None, "torch.distributed unavailable"
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
        self.rank = dist.get_rank()

        if torch.cuda.is_available():
            self._device = torch.device(f"cuda:{self.rank}")
            torch.cuda.set_device(self._device)
        else:
            self._device = torch.device("cpu")

    def dist_wait(self):
        if dist.is_initialized():
            dist.barrier()

    @property
    def is_master(self) -> bool:
        return self.rank == 0

    @property
    def device(self) -> torch.device:
        return self._device

    def cleanup(self):
        if dist.is_initialized():
            dist.destroy_process_group()


class TrainingContextFactory:
    @staticmethod
    def create(cfg: TrainingConfig) -> TrainingContext:
        if cfg.ddp:
            return DistributedTrainingContext(cfg)
        else:
            return SingleTrainingContext(cfg)
