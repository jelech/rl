import argparse

from core.trainer import Trainer
from config.config import Config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ray", action="store_true", help="enable Ray actor-learner")
    p.add_argument("--num_actors", type=int, default=4, help="number of Ray actors")
    p.add_argument("--ddp", action="store_true", help="enable PyTorch DDP")
    p.add_argument("--log_backend", choices=["tensorboard", "mlflow"], default="tensorboard")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.from_json("config.json")
    trainer = Trainer(cfg)
    trainer.train()
