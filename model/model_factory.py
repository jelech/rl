import gymnasium as gym
import torch
import torch.nn as nn

from core.training_context import TrainingContext
from model.model import ActorCritic
from ..config import Config


class ModelFactory:
    @staticmethod
    def create_policy(cfg: Config, context: TrainingContext) -> nn.Module:
        # 创建基础模型
        dummy_env = gym.make(cfg.env.env_id)
        obs_dim = dummy_env.observation_space.shape[0]  # type: ignore
        act_dim = dummy_env.action_space.n  # type: ignore
        policy = ActorCritic(obs_dim, act_dim, cfg.model.hidden_dim).to(context.device)

        if cfg.training.ddp:
            policy = torch.nn.parallel.DistributedDataParallel(
                policy, device_ids=[context.device] if context.device.type == "cuda" else None
            )

        return policy

    @staticmethod
    def get_model_for_saving(model: nn.Module, is_ddp: bool) -> nn.Module:
        return model.module if is_ddp else model

    @staticmethod
    def get_model_for_inference(model: nn.Module) -> nn.Module:
        return model
