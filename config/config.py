import json
import yaml
import os
from typing import Dict, Any
from dataclasses import dataclass, field

from .env_config import EnvConfig
from .training_config import AlgorithmConfig, ModelConfig, TrainingConfig


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_json(cls, json_path: str) -> "Config":
        with open(json_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        # 创建默认配置实例
        config_manager = cls()
        if "env" in config_dict:
            config_manager.env = EnvConfig.from_dict(config_dict["env"])
        if "algorithm" in config_dict:
            config_manager.algorithm = AlgorithmConfig.from_dict(config_dict["algorithm"])
        if "model" in config_dict:
            config_manager.model = ModelConfig.from_dict(config_dict["model"])
        if "training" in config_dict:
            config_manager.training = TrainingConfig.from_dict(config_dict["training"])

        return config_manager

    def to_dict(self) -> Dict[str, Any]:
        return {
            "env": self.env.to_dict(),
            "algorithm": self.algorithm.to_dict(),
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
        }

    def to_json(self, json_path: str) -> None:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    def to_yaml(self, yaml_path: str) -> None:
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f)
