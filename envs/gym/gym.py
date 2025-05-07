import gymnasium as gym
from typing import Any, Dict, SupportsFloat, Tuple, Optional, Union, Type

from envs.env import BaseEnv


class GymEnv(BaseEnv):
    def __init__(self, env_id: str, **kwargs):
        self.env = gym.make(env_id, **kwargs)

    def reset(self) -> Tuple[Any, Dict]:
        return self.env.reset()

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict]:
        return self.env.step(action)

    @property
    def obs_dim(self) -> int:
        if hasattr(self.env.observation_space, "shape"):
            return self.env.observation_space.shape[0]  # type: ignore
        else:
            return self.env.observation_space.n  # type: ignore

    @property
    def action_dim(self) -> int:
        if hasattr(self.env.action_space, "n"):
            return self.env.action_space.n  # type: ignore
        elif hasattr(self.env.action_space, "shape"):
            return self.env.action_space.shape[0]  # type: ignore
        else:
            raise ValueError("Unsupported action space type")

    @property
    def action_type(self) -> str:
        if hasattr(self.env.action_space, "n"):
            return "discrete"
        elif hasattr(self.env.action_space, "shape"):
            return "continuous"
        else:
            return "unknown"
