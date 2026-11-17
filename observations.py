"""Policy boundary: food plus self and one nearest neighbor, never full agent state."""
import numpy as np

OBSERVATION_SIZE = 2508


def distance(a, b, metric: str) -> float:
    delta = np.abs(np.asarray(a) - np.asarray(b))
    return float(delta.sum() if metric == "manhattan" else np.sqrt((delta ** 2).sum()))


class ObservationManager:
    @staticmethod
    def build(environment, identifier: int) -> np.ndarray:
        agent = environment.agents[identifier]
        others = [a for a in environment.agents if a.identifier != identifier]
        distances = [distance(agent.position, a.position, environment.metric) for a in others]
        # Stable tie rule uses only an episode seed, step, and self id, not hidden state.
        candidates = [j for j, d in enumerate(distances) if np.isclose(d, min(distances))]
        tie_rng = np.random.default_rng(np.random.SeedSequence(
            [environment.seed, 61, environment.steps, identifier]))
        neighbor = others[int(tie_rng.choice(candidates))]
        tail = np.asarray([
            agent.position[0] / 49, agent.position[1] / 49,
            neighbor.position[0] / 49, neighbor.position[1] / 49,
            agent.speed / 3,
            float(environment.metric == "manhattan"),
            float(environment.metric == "euclidean"),
            max(0, 1 - environment.steps / environment.config.max_steps),
        ], dtype=np.float32)
        return np.concatenate((environment.board.ravel().astype(np.float32), tail))


def decode(observation: np.ndarray):
    if observation.shape != (OBSERVATION_SIZE,):
        raise ValueError(f"Expected ({OBSERVATION_SIZE},), got {observation.shape}.")
    board = observation[:2500].reshape(50, 50)
    position = np.rint(observation[2500:2502] * 49).astype(int)
    neighbor = np.rint(observation[2502:2504] * 49).astype(int)
    speed = int(round(float(observation[2504]) * 3))
    metric = "manhattan" if observation[2505] > .5 else "euclidean"
    return board, position, neighbor, speed, metric
