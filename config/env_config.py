from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from .base_config import BaseConfig


@dataclass
class EnvConfig(BaseConfig):
    env_id: str = "CartPole-v1"
    env_type: str = "gym"  # "gym", "custom"
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    action_space: str = "auto"  # "discrete", "continuous", "auto"
    observation_space: Optional[Dict[str, Any]] = None
    reward_scale: float = 1.0
    max_episode_steps: Optional[int] = None
