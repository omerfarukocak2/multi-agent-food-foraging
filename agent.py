"""Agent state is owned by the environment, never exposed directly to policies."""
from dataclasses import dataclass
from enum import IntEnum


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    NO_ACTION = 4


DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))


@dataclass
class Agent:
    identifier: int
    position: tuple[int, int]
    speed: int = 1
    green: int = 0
    red: int = 0

    @property
    def score(self) -> int:
        return self.green - self.red
