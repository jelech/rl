from collections import namedtuple
from typing import List, Tuple

Transition = namedtuple("Transition", ["state", "action", "reward", "done", "log_prob", "value"])


class RolloutBuffer:
    def __init__(self):
        self.storage: List[Transition] = []

    def add(self, *args):
        self.storage.append(Transition(*args))

    def extend(self, transitions: List[Tuple]):
        for t in transitions:
            self.add(*t)

    def clear(self):
        self.storage.clear()

    def compute_returns(self, gamma: float):
        returns, R = [], 0.0
        for t in reversed(self.storage):
            R = t.reward + gamma * R * (1 - t.done)
            returns.insert(0, R)
        return returns
