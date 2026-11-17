"""Headless multi-agent training. Run from the project directory."""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from algorithms.rl_agent import RLAgent
from algorithms.features import action_features
from config import Config
from environment import Environment


class TrainingManager:
    def __init__(self, output: Path, episodes: int = 100, steps: int = 500,
                 run: int = 0, validation_maps: int = 3):
        if episodes < 1 or validation_maps < 1:
            raise ValueError("episodes and validation_maps must be positive")
        self.output, self.episodes, self.run = output, episodes, run
        self.config = Config(max_steps=steps)
        self.validation_maps = validation_maps
        self.agents = [RLAgent(7000 + 100 * run + i) for i in range(3)]

    def validate(self):
        env = Environment(self.config)
        scores = []
        for seed in range(500000, 500000 + self.validation_maps):
            obs = env.reset(seed)
            while not env.done:
                obs, _, _, _ = env.step([a.act(o) for a, o in zip(self.agents, obs)])
            scores.append(sum(a.score for a in env.agents))
        return float(np.mean(scores))

    def train(self):
        self.output.mkdir(parents=True, exist_ok=True)
        env = Environment(self.config)
        started, best, history = time.perf_counter(), -float('inf'), []
        settings = {"config": self.config.to_dict(), "episodes": self.episodes,
                    "run": self.run, "validation_maps": self.validation_maps,
                    "learning_rate": .001, "gamma": .9, "batch_size": 64,
                    "replay_capacity_per_agent": 20000, "warmup": 256,
                    "update_every": 4, "target_update_every_gradient_steps": 200,
                    "architecture": "action-conditioned 22-32(tanh)-1 MLP",
                    "epsilon": "max(0.05, 1 - 0.95 * episode / (0.8 * episodes))",
                    "training_seed_start": 10000 + self.run * 100000,
                    "validation_seed_start": 500000, "reward_shaping": False}
        (self.output / 'training_config.json').write_text(json.dumps(settings, indent=2))
        # Episode zero records an untrained validation baseline.
        for episode in range(self.episodes + 1):
            losses = []
            epsilon = max(.05, 1 - .95 * episode / max(1, .8 * self.episodes))
            if episode:
                obs = env.reset(10000 + self.run * 100000 + episode - 1)
                features = [action_features(o) for o in obs]
                while not env.done:
                    actions = [a.choose_features(f, epsilon) for a, f in zip(self.agents, features)]
                    obs, rewards, done, _ = env.step(actions)
                    following = [action_features(o) for o in obs]
                    for i, agent in enumerate(self.agents):
                        agent.remember(features[i], actions[i], rewards[i], following[i], done)
                        loss = agent.learn()
                        if loss is not None:
                            losses.append(loss)
                    features = following
            validation = None
            if episode % 10 == 0 or episode == self.episodes:
                validation = self.validate()
                if validation > best:
                    best = validation
                    for i, agent in enumerate(self.agents):
                        agent.save(self.output / f'agent_{i}.npz')
                    (self.output / 'selection.json').write_text(json.dumps(
                        {"episode": episode, "validation_team_reward": best}, indent=2))
                print(f"run={self.run} episode={episode}/{self.episodes} validation={validation:.1f} best={best:.1f}", flush=True)
            row = {"episode": episode, "epsilon": epsilon,
                   "team_reward": sum(a.score for a in env.agents) if episode else 0,
                   "agent_0_reward": env.agents[0].score if episode else 0,
                   "agent_1_reward": env.agents[1].score if episode else 0,
                   "agent_2_reward": env.agents[2].score if episode else 0,
                   "loss": float(np.mean(losses)) if losses else None,
                   "validation_reward": validation,
                   "elapsed_seconds": time.perf_counter() - started}
            history.append(row)
            with (self.output / 'training.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(history)
        for i, agent in enumerate(self.agents):
            agent.save(self.output / f'last_agent_{i}.npz')
        return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--validation-maps', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('models'))
    args = parser.parse_args()
    if args.runs < 1:
        parser.error('--runs must be positive')
    for run in range(args.runs):
        TrainingManager(args.output / f'run_{run}', args.episodes, args.steps,
                        run, args.validation_maps).train()


if __name__ == '__main__':
    main()
