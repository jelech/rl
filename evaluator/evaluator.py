import gymnasium as gym
import torch
import torch.nn as nn


class Evaluator:
    def __init__(self, env_id: str, policy: nn.Module):
        self.env = gym.make(env_id)
        self.policy = policy

    @torch.no_grad()
    def evaluate(self, episodes: int = 5):
        total = 0.0
        for _ in range(episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                logits, _ = self.policy(state)
                action = torch.argmax(logits).item()
                state, reward, term, trunc, _ = self.env.step(action)
                total += float(reward)
                done = term or trunc
        return total / episodes
