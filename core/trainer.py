from __future__ import annotations


from ..config import Config
from buffer.rollout_buffer import RolloutBuffer
from collector.collector_factory import CollectorFactory
from core.checkpoint import Checkpoint
from core.training_context import TrainingContextFactory
from evaluator.evaluator import Evaluator
from leaner.learner import LearnerFactory
from utils.logger import Logger


class Trainer:
    def __init__(self, cfg: Config):
        # 创建训练上下文
        self.context = TrainingContextFactory.create(cfg.training)
        self.cfg = cfg

        # helpers
        self.buffer = RolloutBuffer()
        self.learner = LearnerFactory.create(cfg, self.context)
        self.evaluator = Evaluator(cfg.env.env_id, self.learner.get_policy_for_inference())
        self.logger = Logger(cfg.training, active=self.context.is_master)
        self.checkpoint = Checkpoint(cfg.training, active=self.context.is_master)

        # 使用工厂创建收集器
        self.collectors, self.is_remote = CollectorFactory.create(
            cfg=cfg,
            policy=self.learner.get_policy_for_inference(),
            buffer=self.buffer,
        )

    def _collect(self):
        return CollectorFactory.collect_data(
            collectors=self.collectors,
            is_remote=self.is_remote,
            policy=self.learner.get_policy_for_inference(),
            buffer=self.buffer,
        )

    def train(self):
        for ep in range(1, self.cfg.training.total_episodes + 1):
            ep_reward = self._collect()
            metrics = self.learner.learn(self.buffer)
            self.buffer.clear()
            metrics["episode_reward"] = ep_reward
            if self.context.is_master:
                self.logger.log(metrics)

            # evaluation / checkpoint
            if self.context.is_master and ep % self.cfg.training.eval_interval == 0:
                avg_r = self.evaluator.evaluate()
                self.logger.log({"eval_avg_reward": avg_r})
                self.checkpoint.save(self.learner.get_policy_for_saving(), ep)
                print(f"Ep {ep:04d} │ TrainR {ep_reward:.1f} │ EvalR {avg_r:.1f}")

        avg_r = self.evaluator.evaluate()
        print(f"Ep finished │ EvalR {avg_r:.1f}")

        # save final checkpoint
        self.checkpoint.close(self.learner.get_policy_for_saving())

        # tear‑down
        self.context.cleanup()
        self.logger.close()
