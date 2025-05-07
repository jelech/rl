from dataclasses import dataclass
from typing import Dict, Any, TypeVar, Type

# 定义泛型类型变量，用于 from_dict 返回类型
T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig:
    @classmethod
    def from_dict(cls: Type[T], config_dict: Dict[str, Any]) -> T:
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in vars(self).items()}
