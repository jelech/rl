from .self_envs.self_envs import SelfEnv
from .env import BaseEnv
from .gym.gym import GymEnv


class EnvFactory:
    @staticmethod
    def create(env_id: str, **kwargs) -> BaseEnv:
        if env_id == "CacheOrder-v0":
            return SelfEnv(env_id, **kwargs)
        else:
            return GymEnv(env_id, **kwargs)
