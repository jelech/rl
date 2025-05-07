import ray

from envs.factory import EnvFactory

from .local_collector import LocalCollector
from .ray_collector import RemoteCollector
from ..config import Config


class CollectorFactory:
    @staticmethod
    def create(cfg: Config, policy, buffer, **env_kwargs):
        env_id = cfg.env.env_id

        if cfg.training.ray:
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)

            collectors = [
                RemoteCollector.remote(env_id, cfg.model.hidden_dim, **env_kwargs)
                for _ in range(cfg.training.num_actors)
            ]
            is_remote = True
        else:
            # 创建环境，支持基本环境或自定义环境
            env = EnvFactory.create(env_id, **env_kwargs)
            collectors = [LocalCollector(env, policy, buffer)]
            is_remote = False

        return collectors, is_remote

    @staticmethod
    def collect_data(collectors, is_remote, policy=None, buffer=None):
        if is_remote:
            if policy is None or buffer is None:
                raise ValueError("Policy and buffer must be provided for remote collection")

            weights = {k: v.detach().cpu().numpy() for k, v in policy.state_dict().items()}
            futures = [collector.collect.remote(weights) for collector in collectors]
            results = ray.get(futures)  # type: ignore

            # 获取远程转换数据并添加到本地缓冲区
            transition_futures = [collector.get_buffer.remote() for collector in collectors]
            all_transitions = ray.get(transition_futures)  # type: ignore
            for transitions in all_transitions:
                buffer.extend(transitions)

            # 计算平均奖励
            ep_rewards = sum(results)
            return ep_rewards / len(results)
        else:
            # 本地收集器直接操作缓冲区
            collector = collectors[0]
            return collector.collect()
