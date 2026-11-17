"""Risk-aware target selection with a lazy reverse Dijkstra cost-to-go planner."""
import heapq
import numpy as np
from algorithms.features import action_features
from observations import decode


class HeuristicAgent:
    name = "Heuristic"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.target = None
        self.previous = None
        self.costs = None
        self.frontier = []
        self.closed = None
        self.red_mask = None

    def _plan(self, board: np.ndarray, endpoints: list[tuple[int, int]]) -> np.ndarray:
        red = board == -1
        if self.costs is None or self.red_mask is None or not np.array_equal(red, self.red_mask):
            self.red_mask = red.copy()
            self.costs = np.full((50, 50), np.inf)
            self.closed = np.zeros((50, 50), dtype=bool)
            self.costs[self.target] = 0
            self.frontier = [(0., *self.target)]
        pending = {p for p in endpoints if not self.closed[p]}
        while pending and self.frontier:
            value, row, col = heapq.heappop(self.frontier)
            if self.closed[row, col]:
                continue
            self.closed[row, col] = True
            pending.discard((row, col))
            # Reverse relaxation: a predecessor pays the cost of entering this cell.
            following = value + 1 + 6 * self.red_mask[row, col]
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = row + dr, col + dc
                if 0 <= rr < 50 and 0 <= cc < 50 and following < self.costs[rr, cc]:
                    self.costs[rr, cc] = following
                    heapq.heappush(self.frontier, (following, rr, cc))
        return self.costs

    def act(self, observation: np.ndarray) -> int:
        board, position, neighbor, speed, metric = decode(observation)
        greens = np.argwhere(board == 1)
        if not len(greens):
            self.target = None
            return 4
        delta = greens - position
        dist = np.abs(delta).sum(axis=1) if metric == "manhattan" else np.linalg.norm(delta, axis=1)
        rival_dist = np.abs(greens - neighbor).sum(axis=1)
        utility = dist + 1.5 * (rival_dist < dist)
        if self.target is None or board[self.target] != 1:
            target = greens[int(np.argmin(utility))]
            self.target = tuple(map(int, target))
            self.costs = None
        f = action_features(observation)
        from agent import DELTAS
        endpoints = [tuple(map(int, np.clip(position + np.array(d) * speed, 0, 49)))
                     for d in DELTAS]
        costs = self._plan(board, endpoints)
        scores = np.zeros(5)
        for action, direction in enumerate(DELTAS):
            end = endpoints[action]
            travel = np.abs(np.asarray(end) - position).sum()
            scores[action] = (3 * f[action, [0, 2, 4]].sum()
                              - 6 * f[action, [1, 3, 5]].sum() - travel - costs[end])
            if self.previous is not None and tuple(end) == self.previous:
                scores[action] -= .05
            if np.array_equal(end, position):
                scores[action] -= .4 if action == 4 else .6
        self.previous = tuple(position)
        best = np.flatnonzero(np.isclose(scores, scores.max()))
        return int(self.rng.choice(best))
