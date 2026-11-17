"""Validated configuration. Coordinates are (row, column).

Fixed contract: uniform speed 1..3 before every decision; one uniform metric
selection per episode; synchronous cell-by-cell paths; no respawn or shaping;
overlap allowed; Left priority on simultaneous green claims. These semantics
are deliberately not independent toggles: changing them requires matching tests.
Full rationale and unresolved instructor interpretations: docs/ASSUMPTIONS.md.
"""
from dataclasses import asdict, dataclass
from numbers import Integral


@dataclass(frozen=True)
class Config:
    size: int = 50
    agents: int = 3
    food_per_color: int = 375
    max_steps: int = 500

    def __post_init__(self) -> None:
        if any(isinstance(v, bool) or not isinstance(v, Integral)
               for v in (self.size, self.agents, self.food_per_color, self.max_steps)):
            raise ValueError("All environment configuration values must be integers.")
        if self.size != 50 or self.agents != 3:
            raise ValueError("The assignment requires a 50x50 board and exactly 3 agents.")
        if not 1 <= self.food_per_color <= 375 or self.max_steps < 1:
            raise ValueError("food_per_color must be 1..375 and max_steps must be positive.")

    def to_dict(self) -> dict:
        return asdict(self)
