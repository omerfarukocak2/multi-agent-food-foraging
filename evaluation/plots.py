import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = ['#8494a7', '#0caa8b', '#7563d7']
LABELS = ['Random', 'Heuristic', 'Double DQN']


def create_plots(rows, summary, runs, output: Path):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'axes.titleweight': 'bold'})
    names = ['random', 'heuristic', 'rl']

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / f'{name}.png', dpi=180)
        fig.savefig(output / f'{name}.pdf')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    values = [summary['algorithms'][name]['mean_team_reward'] for name in names]
    intervals = [summary['algorithms'][name]['team_reward_ci95'] for name in names]
    errors = np.array([[v - ci[0], ci[1] - v] for v, ci in zip(values, intervals)]).T
    ax.bar(LABELS, values, color=COLORS, yerr=errors, capsize=5)
    ax.set(title='Held-out algorithm comparison', ylabel='Mean team reward (95% bootstrap CI)')
    save(fig, 'algorithm_comparison')

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(3)
    for offset, metric, color in [(-.18, 'green', '#0caa8b'), (.18, 'red', '#e16878')]:
        ax.bar(x + offset, [summary['algorithms'][n]['per_agent_metrics'][metric]['mean']
                            for n in names], width=.36, color=color, label=metric.capitalize())
    ax.set(xticks=x, xticklabels=LABELS, ylabel='Mean food collected per agent',
           title='Green and red food collection')
    ax.legend(frameon=False)
    save(fig, 'food_collection')

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, name in enumerate(names):
        selected = [r for r in rows if r['mode'] == name]
        means = [np.mean([r['reward'] for r in selected if r['agent'] == a]) for a in range(3)]
        ax.bar(np.arange(3) + (i - 1) * .24, means, .24, color=COLORS[i], label=LABELS[i])
    ax.set(xticks=np.arange(3), xticklabels=['Agent 1', 'Agent 2', 'Agent 3'],
           ylabel='Mean individual reward', title='Independent agent reward comparison')
    ax.legend(frameon=False)
    save(fig, 'agent_comparison')

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, name in enumerate(names):
        rewards = [np.mean([sum(r['reward'] for r in rows if r['mode'] == name
                                 and r['run'] == run and r['seed'] == summary['test_seeds'][0] + m)
                            for run in range(summary['runs'])]) for m in range(summary['maps'])]
        ax.plot(range(1, len(rewards) + 1), rewards, label=LABELS[i], color=COLORS[i], alpha=.8)
    ax.set(xlabel='Held-out map index', ylabel='Team reward (mean over runs)',
           title='Episode versus total reward')
    ax.legend(frameon=False)
    save(fig, 'episode_reward')

    histories = []
    for run in runs:
        with (run / 'training.csv').open() as stream:
            histories.append(list(csv.DictReader(stream)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    length = min(map(len, histories))
    training = np.array([[float(r['team_reward']) for r in h[:length]] for h in histories])
    xs = np.arange(1, length)
    axes[0].plot(xs, training.mean(axis=0)[1:], color=COLORS[2])
    axes[0].fill_between(xs, training.mean(axis=0)[1:] - training.std(axis=0)[1:],
                         training.mean(axis=0)[1:] + training.std(axis=0)[1:],
                         color=COLORS[2], alpha=.18, label='Across-run +/- 1 SD')
    axes[0].set(title='Exploratory training episodes', xlabel='Episode', ylabel='Team reward')
    axes[0].legend(frameon=False)
    validation = [[(int(r['episode']), float(r['validation_reward'])) for r in h
                   if r['validation_reward']] for h in histories]
    for i, series in enumerate(validation):
        axes[1].plot(*zip(*series), alpha=.65, label=f'Run {i + 1}')
    axes[1].set(title='Fixed validation maps, epsilon = 0', xlabel='Training episode', ylabel='Team reward')
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, 'learning_curve')

    fig, ax = plt.subplots(figsize=(8, 4))
    mixed = [r for r in rows if r['mode'] == 'mixed']
    ax.bar(LABELS, [np.mean([r['reward'] for r in mixed if r['policy'] == name])
                    for name in names], color=COLORS)
    ax.set(title='Mixed-policy competition (rotating starting slots)', ylabel='Mean individual reward')
    save(fig, 'mixed_competition')
