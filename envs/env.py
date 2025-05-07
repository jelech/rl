from abc import ABC, abstractmethod
from typing import Any, Dict, SupportsFloat, Tuple


class BaseEnv(ABC):
    @abstractmethod
    def reset(self) -> Tuple[Any, Dict]:
        raise Exception("reset method not implemented")

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict]:
        raise Exception("reset method not implemented")

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        raise Exception("reset method not implemented")

    @property
    @abstractmethod
    def action_dim(self) -> int:
        raise Exception("reset method not implemented")

    @property
    @abstractmethod
    def action_type(self) -> str:
        raise Exception("reset property not implemented")
