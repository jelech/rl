from .cache_order.cache_order import CacheOrderEnv
from .env import BaseEnv
from .gym.gym import GymEnv


class EnvFactory:
    @staticmethod
    def create(env_id: str, **kwargs) -> BaseEnv:
        if env_id == "CacheOrder-v0":
            return CacheOrderEnv(env_id, **kwargs)
        else:
            return GymEnv(env_id, **kwargs)
