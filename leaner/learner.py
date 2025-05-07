import torch.optim as optim

from algorithm import AlgorithmBase
from algorithm.ppo import PPOAlgorithm
from buffer.rollout_buffer import RolloutBuffer
from config.config import Config
from core.training_context import TrainingContext
from model.model_factory import ModelFactory


class Learner:
    def __init__(self, cfg: Config, algorithm: AlgorithmBase, context: TrainingContext):
        self.cfg = cfg
        self.algorithm = algorithm
        self.context = context

        self.policy = ModelFactory.create_policy(cfg, context)
        self.opt = optim.Adam(
            ModelFactory.get_model_for_saving(self.policy, cfg.training.ddp).parameters(), lr=cfg.algorithm.lr
        )

    def learn(self, buffer: RolloutBuffer):
        metrics = self.algorithm.update_policy(self.policy, self.opt, buffer, self.context)
        self.context.dist_wait()
        return metrics

    def get_policy_for_saving(self):
        return ModelFactory.get_model_for_saving(self.policy, self.cfg.training.ddp)

    def get_policy_for_inference(self):
        return ModelFactory.get_model_for_inference(self.policy)


class LearnerFactory:
    @staticmethod
    def create(cfg: Config, context: TrainingContext, algorithm_type="PPO"):
        if algorithm_type == "PPO":
            algorithm = PPOAlgorithm(cfg.algorithm)
        else:
            raise ValueError(f"不支持的算法类型: {algorithm_type}")

        return Learner(cfg, algorithm, context)
