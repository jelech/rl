import torch.nn as nn
from typing import Dict, Optional, Any
from torch.distributions import Categorical


from . import Collector
from ..buffer.rollout_buffer import RolloutBuffer
from ..envs.env import BaseEnv


class LocalCollector(Collector):
    def __init__(self, env: BaseEnv, policy: nn.Module, buffer: RolloutBuffer):
        self.env = env
        self.policy = policy
        self.buffer = buffer

    def collect(self, policy_weights: Optional[Dict[str, Any]] = None) -> float:
        # 本地收集器忽略policy_weights参数，直接使用self.policy
        return self.collect_episode()

    def get_buffer(self) -> RolloutBuffer:
        return self.buffer

    def collect_episode(self) -> float:
        state, _ = self.env.reset()
        ep_r, done = 0.0, False
        while not done:
            logits, value = self.policy(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            next_state, reward, term, trunc, _ = self.env.step(action.item())
            done_flag = term or trunc
            self.buffer.add(
                state,
                action.item(),
                reward,
                done_flag,
                dist.log_prob(action).detach().cpu(),
                value.item(),
            )
            state, ep_r, done = next_state, ep_r + float(reward), done_flag
        return ep_r
