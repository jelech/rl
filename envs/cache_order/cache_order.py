from typing import Any, Dict, SupportsFloat, Tuple
from envs.env import BaseEnv


class CacheOrderEnv(BaseEnv):
    def __init__(self, env_id: str, **kwargs):
        # Initialize the CacheOrder environment here
        pass

    def reset(self) -> Tuple[Any, Dict]:
        # Implement the reset logic for CacheOrder environment
        raise Exception("reset method not implemented")

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict]:
        # Implement the step logic for CacheOrder environment
        raise Exception("reset method not implemented")

    @property
    def obs_dim(self) -> int:
        # Return the observation dimension for CacheOrder environment
        raise Exception("reset method not implemented")

    @property
    def action_dim(self) -> int:
        # Return the action dimension for CacheOrder environment
        raise Exception("reset method not implemented")

    @property
    def action_type(self) -> str:
        # Return the action type for CacheOrder environment
        raise Exception("reset method not implemented")
