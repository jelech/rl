import torch.nn as nn
import torch
from typing import Dict, List, Tuple, Optional, Any
from torch.distributions import Categorical

from envs.factory import EnvFactory

from . import Collector
from model.model import ActorCritic


try:
    import ray  # type: ignore
except ImportError:  # pragma: no cover
    ray = None

if ray is not None:

    @ray.remote
    class RemoteCollector(Collector):  # noqa: D101
        def __init__(self, env_id: str, hidden: int, **env_kwargs):
            self.env = EnvFactory.create(env_id, **env_kwargs)
            obs_dim = self.env.obs_dim
            act_dim = self.env.action_dim

            self.policy = ActorCritic(obs_dim, act_dim, hidden)
            self.transitions: List[Tuple] = []

        def collect(self, policy_weights: Optional[Dict[str, Any]] = None) -> float:
            if policy_weights is None:
                raise ValueError("RemoteCollector requires policy weights to be provided")
            return self.run_episode(policy_weights)

        def get_buffer(self) -> List[Tuple]:
            return self.transitions

        def run_episode(self, weights: Dict[str, torch.Tensor]) -> float:
            self.policy.load_state_dict({k: torch.tensor(v) for k, v in weights.items()})
            state, _ = self.env.reset()
            self.transitions = []
            ep_r, done = 0.0, False
            while not done:
                logits, value = self.policy(state)
                dist = Categorical(logits=logits)
                action = dist.sample()
                next_state, reward, term, trunc, _ = self.env.step(action.item())
                done_flag = term or trunc
                self.transitions.append(
                    (state, action.item(), reward, done_flag, dist.log_prob(action).item(), value.item())
                )
                state, ep_r, done = next_state, ep_r + float(reward), done_flag
            return ep_r
