import torch
import torch.nn as nn
from torch.distributions import Categorical

from buffer.rollout_buffer import RolloutBuffer
from config import AlgorithmConfig


class PPOAlgorithm(AlgorithmBase):
    def __init__(self, cfg: AlgorithmConfig):
        super().__init__(cfg)
        self.clip_ratio = 0.2
        self.epochs = 10

    def compute_loss(self, states, actions, returns, advantages, old_log_probs, policy):
        logits, values = policy(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(log_probs - old_log_probs)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(ratio * advantages, clip_adv).mean()

        value_loss = nn.functional.mse_loss(values.squeeze(), returns)
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        return loss, {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }

    def prepare_data(self, buffer: RolloutBuffer, context, policy):
        states = torch.tensor([t.state for t in buffer.storage], dtype=torch.float32, device=context.device)
        actions = torch.tensor([t.action for t in buffer.storage], device=context.device)
        returns = torch.tensor(buffer.compute_returns(self.cfg.gamma), device=context.device)

        with torch.no_grad():
            logits, values = policy(states)
            dist = Categorical(logits=logits)
            old_log_probs = dist.log_prob(actions)
            advantages = returns - values.squeeze()

        return states, actions, returns, advantages, old_log_probs

    def update_policy(self, policy, optimizer, buffer, context):
        states, actions, returns, advantages, old_log_probs = self.prepare_data(buffer, context, policy)

        metrics_mean = {}
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = self.compute_loss(states, actions, returns, advantages, old_log_probs, policy)
            loss.backward()
            optimizer.step()

            for k, v in metrics.items():
                if k not in metrics_mean:
                    metrics_mean[k] = 0
                metrics_mean[k] += v / self.epochs

        return metrics_mean
