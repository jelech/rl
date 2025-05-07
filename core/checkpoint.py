import os
import torch
import torch.nn as nn

from ..config import TrainingConfig


class Checkpoint:
    def __init__(self, cfg: TrainingConfig, active: bool = True):
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        self.dir, self.active = cfg.checkpoint_dir, active

    def save(self, policy: nn.Module, ep: int):
        if self.active:
            path = os.path.join(self.dir, f"policy_ep{ep}.pth")
            torch.save(policy.state_dict(), path)

    def close(self, policy: nn.Module):
        path = os.path.join(self.dir, "policy_final.pth")
        torch.save(policy.state_dict(), path)
