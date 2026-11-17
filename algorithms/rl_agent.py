"""Independent Double DQN using NumPy, Huber loss, Adam and target networks.

Q(o,a) is an action-conditioned 22-32-1 MLP. Features use permitted observations.
No heuristic action override, demonstration data, or reward shaping is used.
"""
from pathlib import Path
import numpy as np
from algorithms.features import FEATURE_SIZE, action_features


class Network:
    def __init__(self, rng: np.random.Generator):
        self.params = [rng.normal(0, .15, (FEATURE_SIZE, 32)).astype(np.float32),
                       np.zeros(32, np.float32),
                       rng.normal(0, .15, (32, 1)).astype(np.float32),
                       np.zeros(1, np.float32)]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.updates = 0

    def predict(self, x):
        w, b, v, c = self.params
        return (np.tanh(x @ w + b) @ v + c).squeeze(-1)

    def fit(self, x, targets, learning_rate: float) -> float:
        w, b, v, c = self.params
        h = np.tanh(x @ w + b)
        error = (h @ v + c).ravel() - targets
        derivative = np.clip(error, -1, 1)[:, None] / len(x)
        hidden_gradient = (derivative @ v.T) * (1 - h * h)
        grads = [x.T @ hidden_gradient, hidden_gradient.sum(axis=0),
                 h.T @ derivative, derivative.sum(axis=0)]
        norm = np.sqrt(sum(float((g * g).sum()) for g in grads))
        self.updates += 1
        for index, (p, g) in enumerate(zip(self.params, grads)):
            g = g * min(1., 10 / max(norm, 1e-8))
            self.m[index] = .9 * self.m[index] + .1 * g
            self.v[index] = .999 * self.v[index] + .001 * g * g
            m = self.m[index] / (1 - .9 ** self.updates)
            variance = self.v[index] / (1 - .999 ** self.updates)
            p -= learning_rate * m / (np.sqrt(variance) + 1e-8)
        absolute = np.abs(error)
        return float(np.where(absolute <= 1, .5 * error ** 2, absolute - .5).mean())


class RLAgent:
    name = "Double DQN"
    target = None

    def __init__(self, seed: int = 0, capacity: int = 20000, gamma: float = .9,
                 learning_rate: float = .001):
        self.rng = np.random.default_rng(seed)
        self.online = Network(self.rng)
        self.target_net = Network(self.rng)
        self.sync()
        self.capacity, self.gamma, self.learning_rate = capacity, gamma, learning_rate
        self.x = np.empty((capacity, FEATURE_SIZE), np.float32)
        self.next_x = np.empty((capacity, 5, FEATURE_SIZE), np.float32)
        self.rewards = np.empty(capacity, np.float32)
        self.dones = np.empty(capacity, np.float32)
        self.count = self.cursor = self.transitions = 0

    def sync(self):
        self.target_net.params = [p.copy() for p in self.online.params]

    def choose_features(self, features: np.ndarray, epsilon: float = 0.) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(5))
        return int(np.argmax(self.online.predict(features)))

    def act(self, observation: np.ndarray) -> int:
        return self.choose_features(action_features(observation))

    def remember(self, features, action: int, reward: float, next_features, done: bool):
        j = self.cursor
        self.x[j], self.next_x[j] = features[action], next_features
        self.rewards[j], self.dones[j] = reward, done
        self.cursor = (j + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)
        self.transitions += 1

    def learn(self, batch_size: int = 64):
        if self.count < 256 or self.transitions % 4:
            return None
        indices = self.rng.integers(self.count, size=batch_size)
        following = self.next_x[indices]
        selected = self.online.predict(following).argmax(axis=1)
        q_next = self.target_net.predict(following)[np.arange(batch_size), selected]
        targets = self.rewards[indices] + self.gamma * (1 - self.dones[indices]) * q_next
        loss = self.online.fit(self.x[indices], targets, self.learning_rate)
        if self.online.updates % 200 == 0:
            self.sync()
        return loss

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{f"p{i}": p for i, p in enumerate(self.online.params)},
                            version=np.array([1]), feature_size=np.array([FEATURE_SIZE]),
                            gamma=np.array([self.gamma]), learning_rate=np.array([self.learning_rate]))

    @classmethod
    def load(cls, path: str | Path, seed: int = 0):
        with np.load(path, allow_pickle=False) as archive:
            if int(archive['version'][0]) != 1 or int(archive['feature_size'][0]) != FEATURE_SIZE:
                raise ValueError("Unsupported model version or feature encoder.")
            agent = cls(seed, gamma=float(archive['gamma'][0]),
                        learning_rate=float(archive['learning_rate'][0]))
            for i, p in enumerate(agent.online.params):
                value = archive[f'p{i}']
                if value.shape != p.shape or not np.isfinite(value).all():
                    raise ValueError("Corrupt model weights.")
                p[:] = value
            agent.sync()
        return agent
