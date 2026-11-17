"""Paired held-out tests, individual logs, training/map clustered bootstrap."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from algorithms.random_agent import RandomAgent
from algorithms.heuristic_agent import HeuristicAgent
from algorithms.rl_agent import RLAgent
from config import Config
from environment import Environment


def make_policies(names, seed, model_dir):
    policies = []
    for i, name in enumerate(names):
        if name == 'random':
            policies.append(RandomAgent(seed * 7 + i))
        elif name == 'heuristic':
            policies.append(HeuristicAgent(seed * 7 + i))
        elif name == 'rl':
            policies.append(RLAgent.load(model_dir / f'agent_{i}.npz', seed + i))
        else:
            raise ValueError(f'Unknown policy {name}')
    return policies


def bootstrap_interval(matrix: np.ndarray, seed: int = 42):
    """Crossed resampling of independent training runs and paired map seeds."""
    rng = np.random.default_rng(seed)
    matrix = np.asarray(matrix, dtype=float)
    samples = []
    for _ in range(2000):
        runs = rng.integers(matrix.shape[0], size=matrix.shape[0])
        maps = rng.integers(matrix.shape[1], size=matrix.shape[1])
        samples.append(float(matrix[np.ix_(runs, maps)].mean()))
    return list(map(float, np.percentile(samples, [2.5, 97.5])))


class ExperimentEvaluator:
    def __init__(self, models: Path, output: Path, maps: int = 100, steps: int = 500,
                 seed_start: int = 950000):
        if maps < 2:
            raise ValueError('Use at least two test maps.')
        self.models, self.output, self.maps = models, output, maps
        self.config = Config(max_steps=steps)
        self.seed_start = seed_start

    def evaluate(self):
        self.output.mkdir(parents=True, exist_ok=True)
        runs = sorted(p for p in self.models.glob('run_*') if (p / 'agent_0.npz').exists())
        if not runs:
            raise FileNotFoundError('No trained run directories found. Train first.')
        final_seeds = set(range(self.seed_start, self.seed_start + self.maps))
        for run in runs:
            metadata = json.loads((run / 'training_config.json').read_text())
            training_seeds = set(range(metadata['training_seed_start'],
                                       metadata['training_seed_start'] + metadata['episodes']))
            validation_seeds = set(range(metadata['validation_seed_start'],
                                         metadata['validation_seed_start'] + metadata['validation_maps']))
            if final_seeds & (training_seeds | validation_seeds):
                raise ValueError('Test seeds overlap training or validation maps. Choose fresh seeds.')
        rows = []
        # Baselines are repeated once per independent policy seed run, just like RL.
        for run_index, model_dir in enumerate(runs):
            for algorithm in ['random', 'heuristic', 'rl', 'mixed']:
                env = Environment(self.config)
                for map_index in range(self.maps):
                    seed = self.seed_start + map_index
                    if algorithm == 'mixed':
                        names = list(np.roll(['random', 'heuristic', 'rl'], map_index % 3))
                    else:
                        names = [algorithm] * 3
                    policies = make_policies(names, seed + run_index * 1000000, model_dir)
                    obs = env.reset(seed)
                    while not env.done:
                        obs, _, _, _ = env.step([p.act(o) for p, o in zip(policies, obs)])
                    for agent, name in zip(env.agents, names):
                        rows.append({"mode": algorithm, "policy": name, "run": run_index,
                                     "seed": seed, "agent": agent.identifier,
                                     "metric": env.metric, "reward": agent.score,
                                     "green": agent.green, "red": agent.red,
                                     "green_collection_rate": agent.green / self.config.food_per_color,
                                     "steps": env.steps, "reward_per_step": agent.score / env.steps,
                                     **env.totals})
                    if (map_index + 1) % 10 == 0:
                        print(f"test run={run_index} mode={algorithm} maps={map_index + 1}/{self.maps}", flush=True)
        with (self.output / 'episodes.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {"config": self.config.to_dict(), "maps": self.maps,
                   "runs": len(runs), "test_seeds": [self.seed_start, self.seed_start + self.maps - 1],
                   "ci_method": "2000 crossed bootstrap resamples of runs and maps; agents aggregated within episode",
                   "algorithms": {}}
        reward_matrices = {}
        for name in ['random', 'heuristic', 'rl']:
            selected = [r for r in rows if r['mode'] == name]
            matrix = np.array([[sum(r['reward'] for r in selected if r['run'] == run and
                                   r['seed'] == self.seed_start + m) for m in range(self.maps)]
                               for run in range(len(runs))])
            reward_matrices[name] = matrix
            metrics = {}
            for metric in ['reward', 'green', 'red', 'green_collection_rate', 'steps',
                           'reward_per_step', 'overlaps', 'swaps', 'competitions',
                           'green_competitions', 'left_priority_wins', 'boundary_hits']:
                values = np.array([r[metric] for r in selected], float)
                metrics[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
            summary['algorithms'][name] = {
                "mean_team_reward": float(matrix.mean()),
                "std_team_reward": float(matrix.std(ddof=1)),
                "team_reward_ci95": bootstrap_interval(matrix),
                "per_agent_metrics": {k: v for k, v in metrics.items() if k in
                    ['reward', 'green', 'red', 'green_collection_rate', 'steps', 'reward_per_step']},
                "environment_metrics": {k: v for k, v in metrics.items() if k not in
                    ['reward', 'green', 'red', 'green_collection_rate', 'steps', 'reward_per_step']},
                "mean_team_green_rate": 3 * metrics['green_collection_rate']['mean']}
        summary['paired_differences'] = {}
        for name in ['random', 'heuristic']:
            difference = reward_matrices['rl'] - reward_matrices[name]
            summary['paired_differences'][f'rl_minus_{name}'] = {
                "mean_team_reward_difference": float(difference.mean()),
                "ci95": bootstrap_interval(difference)}
        (self.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        from evaluation.plots import create_plots
        create_plots(rows, summary, runs, self.output)
        print(json.dumps({k: v['mean_team_reward'] for k, v in summary['algorithms'].items()}, indent=2))
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, default=Path('models'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--maps', type=int, default=100)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--seed-start', type=int, default=950000)
    args = parser.parse_args()
    ExperimentEvaluator(args.models, args.output, args.maps, args.steps, args.seed_start).evaluate()


if __name__ == '__main__':
    main()
