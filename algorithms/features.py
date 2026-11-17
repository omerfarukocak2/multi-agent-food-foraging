"""Shared visible-information encoder, not a hand-written RL policy.

Every action is described by food on its predicted path and visible geometry.
No opponent action, speed, target, or hidden position is accessed.
"""
import numpy as np
from agent import DELTAS
from observations import decode

FEATURE_SIZE = 22


def action_features(observation: np.ndarray) -> np.ndarray:
    board, position, neighbor, speed, metric = decode(observation)
    greens = np.argwhere(board == 1)
    features = np.zeros((5, FEATURE_SIZE), dtype=np.float32)
    for action, (dr, dc) in enumerate(DELTAS):
        endpoint = position.copy()
        for substep in range(3):
            if substep < speed and action != 4:
                endpoint = np.clip(endpoint + (dr, dc), 0, 49)
                # A boundary stop or repeated endpoint cannot consume food twice.
                previous = position if substep == 0 else np.clip(
                    position + np.array((dr, dc)) * substep, 0, 49)
                food = board[tuple(endpoint)] if np.any(endpoint != previous) else 0
                features[action, 2 * substep] = float(food == 1)
                features[action, 2 * substep + 1] = float(food == -1)
        if len(greens):
            delta = greens - endpoint
            distances = np.abs(delta).sum(axis=1) if metric == "manhattan" else np.linalg.norm(delta, axis=1)
            target = greens[int(np.argmin(distances))]
            displacement = (target - endpoint) / 49
            features[action, 6:8] = displacement
            features[action, 8] = min(distances) / 98
            features[action, 9] = (np.abs(target - neighbor).sum() -
                                    np.abs(target - endpoint).sum()) / 98
        else:
            features[action, 8] = 1
        features[action, 10:12] = (neighbor - endpoint) / 49
        features[action, 12] = np.abs(endpoint - position).sum() / 3
        features[action, 13] = speed / 3
        features[action, 14] = float(metric == "manhattan")
        features[action, 15:20] = np.eye(5, dtype=np.float32)[action]
        features[action, 20] = len(greens) / 375
        features[action, 21] = observation[-1]
    return features
