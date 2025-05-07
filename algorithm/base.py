from abc import ABC, abstractmethod
from config import AlgorithmConfig


class AlgorithmBase(ABC):
    def __init__(self, cfg: AlgorithmConfig):
        self.cfg = cfg

    @abstractmethod
    def update_policy(self, policy, optimizer, buffer, context):
        raise NotImplementedError("Must be implemented in subclass")
