"""Build the revised report from the existing experiment, without rerunning it.

Run from Group1_AI_Project: python -m report.build_reviewed_report
"""
import csv
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from evaluation.evaluate import bootstrap_interval
from report.build_report import ReportBuilder, REFERENCES


class ReviewedReport(ReportBuilder):
    def __init__(self):
        super().__init__(Path('results'), Path('models'),
                         Path('output/pdf/Project_Report_Revised.pdf'))
        self.styles['BodyCustom'].fontSize = 10.5
        self.styles['BodyCustom'].leading = 14.5
        self.styles['BodyCustom'].spaceAfter = 8
        self.styles['BodyCustom'].textColor = colors.HexColor('#222222')
        self.styles['SectionCustom'].fontSize = 16
        self.styles['SectionCustom'].leading = 20
        self.styles['SectionCustom'].spaceAfter = 11
        self.styles['SectionCustom'].keepWithNext = True
        self.styles['CaptionCustom'].fontSize = 8.5
        self.styles['CaptionCustom'].leading = 11.5

    def make_reviewed_figures(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        destination = self.output.parent / 'figures'
        destination.mkdir(parents=True, exist_ok=True)
        names = ['random', 'heuristic', 'rl']
        labels = ['Random', 'Heuristic', 'Double DQN']
        palette = ['#8494a7', '#0caa8b', '#7563d7']
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                             'axes.spines.top': False, 'axes.spines.right': False,
                             'figure.facecolor': 'white', 'axes.titleweight': 'bold'})
        fig, ax = plt.subplots(figsize=(8, 4))
        for index, name in enumerate(names):
            means = [np.mean([float(r['reward']) for r in self.rows
                              if r['mode'] == name and int(r['agent']) == slot])
                     for slot in range(3)]
            ax.bar(np.arange(3) + (index - 1) * .24, means, .24,
                   color=palette[index], label=labels[index])
        ax.set(xticks=np.arange(3), xticklabels=['Agent 1', 'Agent 2', 'Agent 3'],
               ylabel='Mean individual reward', title='Reward by starting agent slot')
        ax.set_ylim(0, 115)
        ax.legend(frameon=False, loc='upper center', ncol=3)
        fig.tight_layout()
        fig.savefig(destination / 'agent_comparison.png', dpi=220)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 4))
        for index, name in enumerate(names):
            means = [sum(float(r['reward']) for r in self.rows
                         if r['mode'] == name and int(r['seed']) == seed) / self.summary['runs']
                     for seed in range(self.summary['test_seeds'][0], self.summary['test_seeds'][1] + 1)]
            ax.plot(range(1, len(means) + 1), means, color=palette[index], label=labels[index], alpha=.85)
        ax.set(xlabel='Held-out map index', ylabel='Team reward (mean over runs)',
               title='Performance across the 60 held-out maps')
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(destination / 'episode_reward.png', dpi=220)
        plt.close(fig)

    def figure(self, name, caption, width=480):
        if name not in ['agent_comparison', 'episode_reward']:
            return super().figure(name, caption, width)
        original_results = self.results
        try:
            self.results = self.output.parent / 'figures'
            super().figure(name, caption, width)
            self.markdown[-4] = f'![{caption}](../output/pdf/figures/{name}.png)'
        finally:
            self.results = original_results

    def verify_data(self):
        """Recompute reported statistics from the saved agent records."""
        s = self.summary
        assert len(self.rows) == s['runs'] * s['maps'] * 4 * 3
        groups = {}
        for row in self.rows:
            assert int(row['reward']) == int(row['green']) - int(row['red'])
            key = (row['mode'], int(row['run']), int(row['seed']))
            groups.setdefault(key, []).append(row)
        for rows in groups.values():
            assert sorted(int(r['agent']) for r in rows) == [0, 1, 2]
            assert sum(int(r['green']) for r in rows) <= 375
            assert sum(int(r['red']) for r in rows) <= 375
            for metric in ['steps', 'overlaps', 'competitions', 'left_priority_wins']:
                assert len({r[metric] for r in rows}) == 1
        matrices = {}
        for name in ['random', 'heuristic', 'rl']:
            m = np.array([[sum(int(r['reward']) for r in groups[(name, run, seed)])
                           for seed in range(s['test_seeds'][0], s['test_seeds'][1] + 1)]
                          for run in range(s['runs'])])
            matrices[name] = m
            a = s['algorithms'][name]
            assert np.isclose(m.mean(), a['mean_team_reward'])
            assert np.isclose(m.std(ddof=1), a['std_team_reward'])
            assert np.allclose(bootstrap_interval(m), a['team_reward_ci95'])
            rows = [r for r in self.rows if r['mode'] == name]
            for key in ['green', 'red', 'reward_per_step']:
                assert np.isclose(np.mean([float(r[key]) for r in rows]),
                                  a['per_agent_metrics'][key]['mean'])
        for name in ['random', 'heuristic']:
            d = matrices['rl'] - matrices[name]
            saved = s['paired_differences']['rl_minus_' + name]
            assert np.isclose(d.mean(), saved['mean_team_reward_difference'])
            assert np.allclose(bootstrap_interval(d), saved['ci95'])
        for run in self.runs:
            with (run / 'training.csv').open() as stream:
                history = list(csv.DictReader(stream))
            best = max((r for r in history if r['validation_reward']),
                       key=lambda r: float(r['validation_reward']))
            selection = json.loads((run / 'selection.json').read_text())
            assert int(best['episode']) == selection['episode']
            assert np.isclose(float(best['validation_reward']), selection['validation_team_reward'])
            for index in range(3):
                with np.load(run / f'agent_{index}.npz') as archive:
                    assert sum(archive[f'p{i}'].size for i in range(4)) == 769
        self.output.parent.mkdir(parents=True, exist_ok=True)
        (self.output.parent / 'report_checks.json').write_text(json.dumps({
            'agent_records_checked': len(self.rows), 'episodes_checked': len(groups),
            'reward_and_food_accounting': 'passed',
            'team_means_standard_deviations_and_bootstrap_intervals': 'passed',
            'paired_differences_and_intervals': 'passed',
            'checkpoint_selection_and_parameter_counts': 'passed',
            'new_training_or_evaluation_run': False,
        }, indent=2), encoding='utf-8')

    def build(self):
        self.verify_data()
        self.make_reviewed_figures()
        s = self.summary
        alg = s['algorithms']
        labels = {'random': 'Random', 'heuristic': 'Heuristic', 'rl': 'Double DQN'}
        runs, maps, steps = s['runs'], s['maps'], s['config']['max_steps']
        epochs = self.settings['episodes']

        # Cover: retain the institution and course from the supplied report.
        self.story.append(Spacer(1, 22 * mm))
        self.p('Istanbul Ticaret University')
        self.p('Software Engineering')
        self.story.append(Spacer(1, 18 * mm))
        self.story.append(Paragraph('Multi-Agent Food Foraging', ParagraphStyle(
            'ReviewedCover', fontName='Helvetica-Bold', fontSize=29, leading=36,
            textColor=colors.HexColor('#12394b'))))
        self.story.append(Spacer(1, 7 * mm))
        self.p('BIL353 - Artificial Intelligence Term Project')
        self.p('A comparison of Random, Heuristic and Independent Double DQN agents')
        self.story.append(Spacer(1, 15 * mm))
        self.table(['EXPERIMENT', 'SETTING'], [
            ['Board and agents', '50 x 50 grid; three independent agents'],
            ['Initial food', '375 green and 375 red cells'],
            ['Training', f'{runs} runs, each with {epochs} episodes'],
            ['Evaluation', f'{maps} test maps; four policy conditions; {runs} repeats'],
            ['Episode length', f'Up to {steps} decision steps'],
            ['Food reward', '+1 for green; -1 for red'],
        ], [155, 325])
        self.p('This report describes the implementation and the experiment recorded in the '
               'project files. The results below use the saved models, training logs and evaluation data.')
        self.page()

        self.heading('1. Abstract')
        self.p(f'In this project, we compare three ways of controlling agents in a food-foraging '
               f'game: random actions, a hand-written heuristic and independent Double DQN. '
               f'Three agents move on a 50 x 50 grid, collecting green food for a positive reward '
               f'and receiving a penalty for red food. Each agent can see the food map, its own '
               f'state and the position of its nearest neighbor. We trained {runs} sets of three '
               f'RL agents for {epochs} episodes per set, then evaluated the methods on {maps} '
               f'new maps with a {steps}-step limit. When all three agents used the same method, '
               f'the average team rewards were {alg["random"]["mean_team_reward"]:.2f} for Random, '
               f'{alg["heuristic"]["mean_team_reward"]:.2f} for Heuristic and '
               f'{alg["rl"]["mean_team_reward"]:.2f} for Double DQN. The learned agents performed '
               'much better than Random, but the heuristic collected food more reliably in this experiment.')
        self.heading('2. Introduction')
        self.p('The aim of the project is to study how different decision methods behave in the '
               'same environment. The rules are simple, but an agent still has to decide whether '
               'to approach nearby food, avoid a red cell or move toward an area that another agent may reach first.')
        self.p('We use Random as a basic comparison, Heuristic as a rule-based solution and '
               'Double DQN as a method that learns from rewards. Following the reinforcement '
               'learning framework in Sutton and Barto [1], an RL agent improves its action '
               'values through experience. The main question here is how well that approach '
               'works against a practical heuristic within a small training budget.')
        self.heading('3. Problem Definition')
        self.p('The board contains 2,500 cells. At the start, 375 cells contain green food and '
               '375 contain red food, so each color occupies 15% of the board. Each agent chooses '
               'Up, Down, Left, Right or No Action. Its speed is sampled uniformly from 1, 2 or 3 '
               'before each decision. A move follows a straight path and can collect food from '
               'intermediate cells as well as the endpoint.')
        self.formula('r_i,t = green collected by agent i - red collected by agent i\n'
                     'R_i = sum_t r_i,t          R_team = R_0 + R_1 + R_2')
        self.p('The agents receive their own rewards. We add their episode scores to compare '
               'team performance and select model checkpoints. An agent cannot observe the '
               'other two agents in full, so its observation does not describe the complete game state.')
        self.page()

        self.heading('4. Related Work')
        self.p('Sutton and Barto [1] explain action values, exploration and learning from rewards. '
               'These ideas provide the starting point for our RL agent. A direct table of all '
               'possible food-map states would be too large, so we approximate action values with a small neural network.')
        self.p('Mnih et al. [2] combine a neural network with experience replay and a target '
               'network in DQN. We use these two training mechanisms, but our input is a set of '
               'numerical features rather than game images. Double DQN [3] uses the online '
               'network to select the next action and the target network to evaluate it. This '
               'separation is intended to reduce overestimated action values.')
        self.p('Tan [4] compares independent learning with approaches that share information '
               'between agents. Our agents learn separately and do not exchange rewards or '
               'replay data. Henderson et al. [5] discuss how random seeds and experimental '
               'choices affect RL results. We therefore report several training runs and confidence intervals.')
        self.heading('5. Methodology')
        self.p('All three methods receive the same observation format. Random chooses each of '
               'the five actions with equal probability. Double DQN scores the five actions '
               'using its learned network; during training it also explores with random actions.')
        self.p('Heuristic selects a green target using its distance plus a penalty of 1.5 '
               'when the visible neighbor is judged closer. It keeps that target until the '
               'food disappears. The agent-to-target distance uses the episode metric, while '
               'the neighbor-to-target distance always uses Manhattan distance. This is an '
               'approximation when the episode metric is Euclidean.')
        self.p('A reverse Dijkstra search estimates the cost of reaching the target. Entering '
               'a red cell costs more than entering an empty or green cell. The action score '
               'also considers food on the immediate path and discourages waiting or returning to the previous position.')
        self.formula('edge_cost(u -> v) = 1 + 6 * I(v contains red food)\n'
                     'score(a) = 3 * green_on_path - 6 * red_on_path\n'
                     '           - path_length - cost_to_target(endpoint)\n'
                     '           - repeat_penalty - stationary_penalty')
        self.p('The repeat penalty is 0.05; the stationary penalty is 0.4 for No Action or '
               '0.6 for a move that cannot leave the cell. These scores only guide decisions. '
               'The actual food rewards remain +1 and -1. The earlier heuristic had difficulty '
               'with red barriers; the separate exploratory results use seeds 900000-900059. '
               'The final comparison uses new test seeds after the path-planning revision.')
        self.page()

        self.heading('6. Environment and Agent Design')
        self.table(['COMPONENT', 'ROLE'], [
            ['Environment', 'Creates maps, supplies observations and advances each decision step'],
            ['MovementResolver', 'Moves agents in substeps and resolves simultaneous food claims'],
            ['Agent / Food', 'Stores agent state and represents food on the board'],
            ['RewardManager', 'Updates food counts and individual rewards'],
            ['ObservationManager', 'Builds the permitted observation for each agent'],
            ['TrainingManager', 'Trains separate agents and selects checkpoints by validation'],
            ['ExperimentEvaluator', 'Runs comparisons and saves per-agent records'],
            ['SimulationRenderer', 'Displays the board, scores and interactive controls'],
        ], [137, 343])
        self.p('Each observation has 2,508 float32 values: 2,500 food cells, two coordinates '
               'for the agent, two for the nearest neighbor, the current speed, two distance-metric '
               'indicators and the fraction of time remaining. All food is visible. The other '
               'agents are absent from the board encoding; neighbor speed, identity, target and score are not included.')
        self.p('At reset, the environment chooses Manhattan or Euclidean distance with equal '
               'probability and uses it for the whole episode. Equal-distance neighbor ties '
               'use a seeded random choice. The agents start in distinct cells without food. '
               'Separate seeded random generators handle maps, distance metrics, speeds and food contests.')
        self.p('Actions are simultaneous, but movement is resolved one cell at a time. Borders '
               'stop movement without a reward penalty, and agents can overlap or swap places. '
               'If several agents claim green food in the same substep, a Left action has '
               'priority. Remaining ties and red-food contests use seeded randomness. Earlier '
               'arrival wins over later arrival, and each food item is consumed once.')
        self.p('Food does not respawn. An episode ends when all food is gone or the time limit '
               'is reached. These choices, including speed timing and overlap behavior, are '
               'listed in docs/ASSUMPTIONS.md as implementation assumptions.')
        test_root = ET.parse(self.results / 'test_results.xml').getroot()
        suites = [test_root] if test_root.tag == 'testsuite' else test_root.findall('testsuite')
        tests = sum(int(t.get('tests', 0)) for t in suites)
        failures = sum(int(t.get('failures', 0)) + int(t.get('errors', 0)) for t in suites)
        self.p(f'The saved test record contains {tests} tests and {failures} failures or errors. '
               'It covers movement, reward accounting, food conservation, information hiding, '
               'model loading, learning updates and renderer controls. This is the recorded '
               'verification run supplied with the experiment.')
        self.page()

        self.heading('7. Reinforcement Learning Algorithm')
        self.p('Each RL agent owns a network, a target network and a replay buffer. The network '
               'takes 22 features for one candidate action and outputs its estimated value. '
               'It has 32 tanh hidden units and one output: 22 x 32 + 32 + 32 + 1 = 769 '
               'trainable parameters per agent. The same network is applied to all five actions.')
        self.p('Six features indicate green or red food on up to three path cells. The remaining '
               'features describe the closest currently visible green from the proposed endpoint, '
               'neighbor geometry, movement length, speed, metric, action identity, green-food '
               'fraction and remaining time. The encoder uses the current board: it does not '
               'first remove food that the proposed move would consume. Its neighbor-to-target '
               'comparison uses Manhattan distance even in Euclidean episodes.')
        self.p('These features give the learner useful spatial information, but they compress '
               'the full observation. No demonstration actions or heuristic overrides are used. '
               'The training target follows Double DQN [3]:')
        self.formula('a* = argmax_b Q_online(o_next, b)\n'
                     'y = r + gamma * (1 - done) * Q_target(o_next, a*)\n'
                     'e = Q_online(o, a) - y\n'
                     'Huber(e) = 0.5 * e^2       if |e| <= 1\n'
                     '           |e| - 0.5      otherwise')
        self.table(['SETTING', 'VALUE'], [
            ['Discount / learning rate', '0.90 / 0.001'],
            ['Replay capacity / warm-up / batch', '20,000 per agent / 256 transitions / 64'],
            ['Update / target copy', 'Every 4 transitions / every 200 gradient updates'],
            ['Optimizer / gradient clipping', 'Adam (0.9, 0.999, epsilon=1e-8) / global norm 10'],
            ['Exploration schedule', 'epsilon = max(0.05, 1 - 0.95 * episode / (0.8 * E))'],
            ['Recorded E / evaluation epsilon', f'{epochs} training episodes / 0'],
        ], [180, 300])
        self.p('Here episode is one-based for training. With E=60, epsilon starts at about '
               '0.9802 and reaches 0.05 at episode 48. The episode-zero row records validation '
               'before training. At the time limit, done=1 and the target contains no future '
               'reward beyond the episode. Saved NPZ files contain inference weights and '
               'metadata; they do not store the optimizer or replay state needed to resume training exactly.')
        self.page()

        self.heading('8. Experimental Setup')
        self.p(f'We trained {runs} independent sets of three agents. Each run used {epochs} '
               f'episodes with at most {steps} decisions per episode. Before training and every '
               'ten episodes, the agents were evaluated without exploration on three fixed '
               'validation maps. The highest validation team reward selected a set of three '
               'checkpoints for each run.')
        self.table(['ITEM', 'SETTING'], [
            ['Training seeds', '10000 + 100000 * run + zero-based episode index'],
            ['Validation seeds', '500000-500002; reused for checkpoint selection'],
            ['Final test seeds', '950000-950059; separate from training and validation'],
            ['Same-method teams', 'Three Random, three Heuristic or three Double DQN agents'],
            ['Mixed team', 'One agent per method; starting slots rotate every map'],
            ['Evaluation size', '4 conditions x 5 repeats x 60 maps = 1,200 episodes'],
            ['Evaluation records', '3 agent records per episode = 3,600 rows'],
            ['Confidence intervals', '2,000 bootstrap resamples of both run and map axes'],
        ], [150, 330])
        self.p('The five repeats correspond to independently trained networks for RL and '
               'different policy random seeds for the baselines. Every condition uses the '
               'same 60 test maps. Map layout, distance metric and the speed sequence for each '
               'starting slot are fixed by the map seed. Policy decisions can still produce '
               'different food contests. In mixed teams, the RL model is the checkpoint '
               'associated with the slot occupied by the RL agent.')
        self.p('The main outcome is team reward in the same-method teams. We first add the '
               'three agent scores for each episode, then form a 5 x 60 matrix. Confidence '
               'intervals resample runs and maps separately and take the 2.5th and 97.5th '
               'percentiles of the resulting means. Comparisons use RL-minus-baseline '
               'differences at matching run and map positions. The three agents in one episode '
               'are not treated as independent samples.')
        self.p('Food counts and reward per step are also reported per agent. Team green '
               'collection is the total green count divided by 375. Contests and overlaps '
               'are environment events copied into each agent row, so they are counted once '
               'per episode. An overlap is counted for each pair at each substep, including stationary overlap.')
        self.p('The saved runtime record specifies Python 3.13.1 and NumPy on CPU. Package '
               'versions are in requirements-lock.txt. The interface defaults to 500 steps, '
               'while this experiment uses 300. New test maps come from the same map generator '
               'as training, so this evaluation measures performance on new seeds under the same rules.')
        self.page()

        self.heading('9. Results and Evaluation')
        self.p('9.1 Same-method teams')
        result_rows = []
        for name, label in labels.items():
            a = alg[name]
            ci = a['team_reward_ci95']
            result_rows.append([label, f'{a["mean_team_reward"]:.2f}',
                                f'{a["std_team_reward"]:.2f}', f'[{ci[0]:.2f}, {ci[1]:.2f}]'])
        self.table(['METHOD', 'TEAM MEAN', 'TEAM SD', '95% CI OF MEAN'], result_rows,
                   [110, 105, 95, 170])
        self.p('Heuristic has the highest mean team score. Double DQN also earns a strong '
               'positive score, but its results vary more across runs and maps. Random is '
               'close to zero because its green gains and red penalties largely cancel out.')
        self.figure('algorithm_comparison',
                    'Figure 1. Mean team reward across 300 episodes per method '
                    '(5 repeats x 60 maps). Error bars show 95% bootstrap intervals.', width=470)
        for name in ['random', 'heuristic']:
            diff = s['paired_differences']['rl_minus_' + name]
            ci = diff['ci95']
            self.p(f'The paired Double DQN minus {labels[name]} difference is '
                   f'{diff["mean_team_reward_difference"]:.2f} team reward, with a 95% '
                   f'interval of [{ci[0]:.2f}, {ci[1]:.2f}].')
        self.p('The RL-minus-Heuristic interval lies below zero, supporting the heuristic '
               'advantage at this training and test budget. SD describes variation between '
               'episode outcomes; the confidence interval describes uncertainty in the mean.')
        self.page()

        self.heading('9.2 Food Collection and Individual Agents')
        self.figure('food_collection',
                    'Figure 2. Average green and red food collected by one agent per episode.', width=420)
        self.figure('agent_comparison',
                    'Figure 3. Average reward by starting slot in same-method teams. '
                    'Agent 1-3 in the figure correspond to code slots 0-2.', width=420)
        food_rows = []
        for name, label in labels.items():
            a = alg[name]
            p = a['per_agent_metrics']
            food_rows.append([label, f'{p["green"]["mean"]:.2f}', f'{p["red"]["mean"]:.2f}',
                              f'{100 * a["mean_team_green_rate"]:.1f}%',
                              f'{p["reward_per_step"]["mean"]:.4f}'])
        self.table(['METHOD', 'GREEN / AGENT', 'RED / AGENT', 'TEAM GREEN %', 'REWARD / AGENT / STEP'],
                   food_rows, [90, 95, 95, 95, 105])
        self.p('Heuristic collects slightly more green food than Double DQN while consuming '
               'fewer red cells. Both differences contribute to its higher score. The food '
               'counts and reward-per-step column refer to one agent; the green percentage '
               'refers to all three agents together.')
        self.page()

        self.heading('9.3 Learning Progress and New Maps')
        self.figure('learning_curve',
                    'Figure 4. Training reward and greedy validation scores. The shaded '
                    'training band shows one SD across five runs. Plot Run 1-5 correspond to run_0-run_4.', width=470)
        selection_rows = []
        for run in self.runs:
            selection = json.loads((run / 'selection.json').read_text())
            with (run / 'training.csv').open() as stream:
                history = list(csv.DictReader(stream))
            selection_rows.append([run.name, f'{float(history[0]["validation_reward"]):.2f}',
                                   selection['episode'], f'{selection["validation_team_reward"]:.2f}'])
        self.table(['RUN', 'INITIAL VALIDATION', 'SELECTED EPISODE', 'SELECTED VALIDATION'],
                   selection_rows, [80, 130, 130, 140])
        self.p('All five selected models score above their untrained validation baselines. '
               'The best checkpoint is sometimes earlier than episode 60, which shows that '
               'more training did not improve every run steadily. These are scores on the '
               'three reused validation maps, rather than final-test results.')
        self.figure('episode_reward',
                    'Figure 5. Team reward on each of the 60 test maps, averaged over five repeats. '
                    'The horizontal axis indexes test maps, not training progress.', width=440)
        self.page()

        self.heading('9.4 Mixed Teams and Agent Interactions')
        self.figure('mixed_competition',
                    'Figure 6. Mean individual reward in mixed teams, with 300 episodes per policy '
                    'and starting slots rotated across maps.', width=450)
        mixed = [r for r in self.rows if r['mode'] == 'mixed']
        mixed_rows = []
        for name, label in labels.items():
            selected = [r for r in mixed if r['policy'] == name]
            mixed_rows.append([label] + [f'{np.mean([float(r[k]) for r in selected]):.2f}'
                                        for k in ['reward', 'green', 'red']])
        self.table(['POLICY', 'REWARD / AGENT', 'GREEN / AGENT', 'RED / AGENT'],
                   mixed_rows, [120, 120, 120, 120])
        self.p('Double DQN has a slightly higher descriptive mean than Heuristic in mixed '
               'teams (89.82 versus 86.38). This table does not include a paired uncertainty '
               'estimate, so the small gap alone is not enough to establish an advantage. '
               'Mixed teams also expose RL to different opponents from those used in training.')
        event_rows = []
        for name, label in labels.items():
            e = alg[name]['environment_metrics']
            event_rows.append([label, f'{e["competitions"]["mean"]:.3f}',
                               f'{e["left_priority_wins"]["mean"]:.3f}',
                               f'{e["overlaps"]["mean"]:.2f}',
                               f'{alg[name]["per_agent_metrics"]["steps"]["mean"]:.0f}'])
        self.table(['SAME-METHOD TEAM', 'CONTESTS', 'LEFT-PRIORITY EVENTS', 'OVERLAPS', 'STEPS'],
                   event_rows, [115, 80, 120, 85, 80])
        self.p('The interaction table returns to same-method teams. Values are means per '
               'environment episode. A Left-priority event is a contested green cell where '
               'at least one contender chose Left. It is not a win rate. Such contests are '
               'rare in the recorded maps; controlled movement tests check the priority rule directly.')
        self.page()

        self.heading('10. Discussion')
        self.p('The clearest result is that Double DQN learns useful behavior, but the '
               'heuristic performs better in the main comparison. Heuristic earns 259.74 '
               'team reward on average, compared with 234.04 for Double DQN. Its lower red '
               'food count and smaller spread in episode scores help explain this difference.')
        self.p('A likely reason is that the full food map is visible. The heuristic can '
               'plan a route to a specific target and account for red cells along that route. '
               'The RL agent instead learns from a compact description of each possible move. '
               'It has useful local information, but the feature vector does not retain every '
               'detail of the map. This is an interpretation of the design and results; '
               'separate feature-removal experiments would be needed to test it.')
        self.p('The mixed-team ranking also shows why the evaluation setting matters. A '
               'method that collects the most food in a team of similar agents may behave '
               'differently when competing with other strategies. The homogeneous team '
               'scores and mixed individual scores should therefore be read separately.')
        self.heading('11. Limitations')
        self.p('The experiment uses five training runs, only three validation maps, one '
               'food density and a 300-step horizon. Reusing a small validation set can '
               'favor checkpoints that suit those maps. The final test set is separate, '
               'but it comes from the same generator, so it does not test changed rules or unfamiliar map types.')
        self.p('We did not compare Double DQN with ordinary DQN or remove features one at '
               'a time. The results therefore cannot isolate the benefit of the Double DQN '
               'target or any particular feature. The agents also observe only one neighbor, '
               'and their learning opponents change during training. These factors make '
               'learning less stable. Longer training, more validation maps and additional '
               'random seeds would give a fuller picture.')
        self.p('Saved models support inference but not exact mid-training resumption. '
               'The interface has a different default horizon, and renderer tests do not '
               'cover every desktop setup. The rule assumptions should also be checked '
               'against the instructor\'s final requirements.')
        self.heading('12. Conclusion')
        self.p('The project brings together a working multi-agent environment, three '
               'decision methods and a visual interface. Double DQN improves substantially '
               'over Random, while the path-based heuristic remains the stronger method '
               'for same-method teams in this experiment. A useful next step is to expand '
               'validation and training, then compare ordinary DQN and simpler feature '
               'sets under the same test conditions.')
        self.page()

        self.heading('13. References')
        refs = list(REFERENCES)
        refs[0] = (refs[0][0], 'https://mitpress.mit.edu/9780262039246/reinforcement-learning/')
        for index, (description, url) in enumerate(refs, 1):
            self.p(f'[{index}] {description}')
            self.story.append(Paragraph(f'<link href="{html.escape(url)}" color="#12394b">'
                                        f'{html.escape(url)}</link>', self.styles['CaptionCustom']))
            self.markdown.extend([url, ''])
        self.story.append(Spacer(1, 8 * mm))
        self.p('Project files used for the results')
        self.p('The numerical results come from results/episodes.csv and results/summary.json. '
               'Training progress and selected checkpoints are recorded in models/run_*/training.csv '
               'and selection.json. The rule assumptions are in docs/ASSUMPTIONS.md, and the '
               'saved automated-test record is results/test_results.xml.')
        self.p('README.md gives installation, training and evaluation commands. This revised '
               'report can be rebuilt from the project directory with the command below. '
               'It checks the saved statistics and does not start a new training run.')
        self.formula('python -m report.build_reviewed_report')

        self.output.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(str(self.output), pagesize=(210 * mm, 297 * mm),
                                rightMargin=20 * mm, leftMargin=20 * mm,
                                topMargin=21 * mm, bottomMargin=20 * mm,
                                title='Multi-Agent Food Foraging - Project Report',
                                author='BIL353 Project', pageCompression=1)

        def decorate(canvas, document):
            canvas.saveState()
            canvas.setFont('Helvetica', 8)
            canvas.setFillColor(colors.HexColor('#58636e'))
            if document.page > 1:
                canvas.drawString(20 * mm, 286 * mm, 'BIL353 | Multi-Agent Food Foraging')
                canvas.setStrokeColor(colors.HexColor('#d0d6dc'))
                canvas.line(20 * mm, 282 * mm, 190 * mm, 282 * mm)
            canvas.drawString(20 * mm, 12 * mm, 'Artificial Intelligence Term Project')
            canvas.drawRightString(190 * mm, 12 * mm, str(document.page))
            canvas.restoreState()

        doc.build(self.story, onFirstPage=decorate, onLaterPages=decorate)
        (Path('report') / 'Project_Report_Revised.md').write_text(
            '\n'.join(self.markdown), encoding='utf-8')
        print(f'Built {self.output}; all numerical checks passed.')


if __name__ == '__main__':
    ReviewedReport().build()
