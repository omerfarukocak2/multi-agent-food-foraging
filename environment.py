"""Assignment-compatible environment with explicit, reproducible stochastic rules."""
import numbers
import numpy as np
from agent import Agent
from config import Config
from movement import MovementResolver
from observations import ObservationManager


class Environment:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.reset(0)

    def reset(self, seed: int = 0) -> list[np.ndarray]:
        if not isinstance(seed, numbers.Integral) or seed < 0:
            raise ValueError("Seed must be a nonnegative integer.")
        self.seed = int(seed)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 11]))
        cells = rng.permutation(2500)
        self.board = np.zeros((50, 50), dtype=np.int8)
        count = self.config.food_per_color
        self.board.flat[cells[:count]] = 1
        self.board.flat[cells[count:2 * count]] = -1
        self.agents = [Agent(i, tuple(map(int, np.unravel_index(cells[2 * count + i],
                                                                           (50, 50)))))
                       for i in range(3)]
        metric_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 21]))
        self.metric = str(metric_rng.choice(["manhattan", "euclidean"]))
        self.steps = 0
        self.done = False
        self.totals = {name: 0 for name in ["overlaps", "swaps", "competitions",
                       "green_competitions", "left_priority_wins", "boundary_hits"]}
        self._sample_speeds()
        return self.observe()

    def _sample_speeds(self) -> None:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 31, self.steps]))
        for agent, speed in zip(self.agents, rng.integers(1, 4, size=3)):
            agent.speed = int(speed)

    def observe(self) -> list[np.ndarray]:
        return [ObservationManager.build(self, i) for i in range(3)]

    def step(self, actions: list[int]):
        if self.done:
            raise RuntimeError("Episode finished. Call reset before step.")
        if len(actions) != 3 or any(isinstance(a, bool) or not isinstance(a, numbers.Integral)
                                    or not 0 <= a <= 4 for a in actions):
            raise ValueError("Exactly three integer actions in 0..4 are required.")
        rewards, events, paths = MovementResolver.resolve(self, list(map(int, actions)))
        for key, value in events.items():
            self.totals[key] += value
        self.steps += 1
        terminated = not np.any(self.board)
        truncated = self.steps >= self.config.max_steps and not terminated
        self.done = terminated or truncated
        self._sample_speeds()
        return self.observe(), rewards, self.done, {
            "terminated": terminated, "truncated": truncated,
            "events": events, "paths": paths, "metric": self.metric}
