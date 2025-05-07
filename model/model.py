import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, act_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
        x = self.shared(x)
        return self.policy_head(x), self.value_head(x)
