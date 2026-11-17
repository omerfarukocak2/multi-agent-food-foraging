import numpy as np


class RandomAgent:
    name = "Random"
    target = None

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray) -> int:
        return int(self.rng.integers(5))
