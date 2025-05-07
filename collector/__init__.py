from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List, Tuple


class Collector(ABC):
    @abstractmethod
    def collect(self, policy_weights: Optional[Dict[str, Any]] = None) -> float:
        """收集数据的方法，返回该回合的累积奖励"""
        pass

    @abstractmethod
    def get_buffer(self) -> Union[Any, List[Tuple]]:
        """获取收集的数据，可能是缓冲区对象或转换列表"""
        pass


from .local_collector import LocalCollector
from .collector_factory import CollectorFactory

# 公开的模块API
__all__ = ["Collector", "LocalCollector", "CollectorFactory"]
