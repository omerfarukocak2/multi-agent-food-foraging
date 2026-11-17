"""Build the English academic report from actual, recorded experimental outputs."""
import argparse
import csv
import html
import json
from pathlib import Path
import platform
import xml.etree.ElementTree as ET
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                               TableStyle, Image, PageBreak, KeepTogether)

REFERENCES = [
    ('Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction '
     '(2nd ed.). MIT Press.', 'https://mitpress.ublish.com/book/reinforcement-learning-an-introduction-2'),
    ('Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. '
     'Nature, 518, 529-533. doi:10.1038/nature14236.', 'https://www.nature.com/articles/nature14236'),
    ('van Hasselt, H., Guez, A., & Silver, D. (2015). Deep Reinforcement Learning '
     'with Double Q-learning. arXiv:1509.06461; subsequently AAAI 2016.',
     'https://arxiv.org/abs/1509.06461'),
    ('Tan, M. (1993). Multi-Agent Reinforcement Learning: Independent Versus '
     'Cooperative Agents. ICML, 330-337. doi:10.1016/B978-1-55860-307-3.50049-6.',
     'https://mlanthology.org/icml/1993/tan1993icml-multi/'),
    ('Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D. '
     '(2017). Deep Reinforcement Learning that Matters. arXiv:1709.06560; '
     'subsequently AAAI 2018.', 'https://arxiv.org/abs/1709.06560'),
]


class ReportBuilder:
    def __init__(self, results: Path, models: Path, output: Path):
        self.results, self.models, self.output = results, models, output
        self.summary = json.loads((results / 'summary.json').read_text())
        with (results / 'episodes.csv').open() as stream:
            self.rows = list(csv.DictReader(stream))
        self.runs = sorted(p for p in models.glob('run_*') if (p / 'training.csv').exists())
        self.settings = json.loads((self.runs[0] / 'training_config.json').read_text())
        self.story, self.markdown = [], ['# Multi-Agent Food Foraging', '',
                                        'BIL353 - Artificial Intelligence Term Project', '']
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle('BodyCustom', fontName='Helvetica', fontSize=10,
                                     leading=14.7, textColor=colors.HexColor('#24354a'), spaceAfter=9))
        self.styles.add(ParagraphStyle('SectionCustom', fontName='Helvetica-Bold', fontSize=17,
                                     leading=22, textColor=colors.HexColor('#12394b'), spaceAfter=13))
        self.styles.add(ParagraphStyle('CaptionCustom', fontName='Helvetica', fontSize=8.5,
                                     leading=12, textColor=colors.HexColor('#62718a'), spaceAfter=10))
        self.styles.add(ParagraphStyle('CellCustom', fontName='Helvetica', fontSize=8.3, leading=11))
        self.styles.add(ParagraphStyle('FormulaCustom', fontName='Courier', fontSize=8.3,
                                     leading=13, backColor=colors.HexColor('#edf3f8'),
                                     borderPadding=9, spaceBefore=5, spaceAfter=14))

    def p(self, text):
        self.story.append(Paragraph(html.escape(text), self.styles['BodyCustom']))
        self.markdown.extend([text, ''])

    def heading(self, text):
        self.story.append(Paragraph(html.escape(text), self.styles['SectionCustom']))
        self.markdown.extend(['## ' + text, ''])

    def page(self):
        self.story.append(PageBreak())

    def formula(self, text):
        self.story.append(Paragraph(html.escape(text).replace('\n', '<br/>'), self.styles['FormulaCustom']))
        self.markdown.extend(['```text', text, '```', ''])

    def table(self, headers, rows, widths=None):
        data = [[Paragraph(html.escape(str(cell)), self.styles['CellCustom']) for cell in row]
                for row in [headers, *rows]]
        table = Table(data, colWidths=widths or [480 / len(headers)] * len(headers), repeatRows=1,
                      hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dcebf1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f4f7fa')]),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('LINEBELOW', (0, 0), (-1, 0), .7, colors.HexColor('#adc6d1')),
        ]))
        self.story.extend([table, Spacer(1, 12)])
        self.markdown.extend(['| ' + ' | '.join(map(str, headers)) + ' |',
                              '| ' + ' | '.join(['---'] * len(headers)) + ' |'])
        self.markdown.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)
        self.markdown.append('')

    def figure(self, name, caption, width=480):
        path = self.results / (name + '.png')
        from PIL import Image as PILImage
        with PILImage.open(path) as raster:
            height = width * raster.height / raster.width
        self.story.append(KeepTogether([Image(str(path), width=width, height=height),
                                        Paragraph(html.escape(caption), self.styles['CaptionCustom'])]))
        self.markdown.extend([f'![{caption}](../{self.results.name}/{name}.png)', '', caption, ''])

    def build(self):
        s = self.summary
        algorithms = s['algorithms']
        runs, maps, horizon = s['runs'], s['maps'], s['config']['max_steps']
        episodes = self.settings['episodes']
        rr = algorithms['rl']['mean_team_reward']
        hr = algorithms['heuristic']['mean_team_reward']
        random_reward = algorithms['random']['mean_team_reward']
        tests = None
        test_path = self.results / 'test_results.xml'
        if test_path.exists():
            test_root = ET.parse(test_path).getroot()
            suites = test_root.findall('testsuite') if test_root.tag != 'testsuite' else [test_root]
            tests = sum(int(t.attrib.get('tests', 0)) for t in suites)
        self.story.append(Spacer(1, 34 * mm))
        self.story.append(Paragraph('MULTI-AGENT<br/>FOOD FORAGING', ParagraphStyle(
            'Cover', fontName='Helvetica-Bold', fontSize=34, leading=39,
            textColor=colors.HexColor('#12394b'))))
        self.story.append(Spacer(1, 15 * mm))
        self.p('Independent Double DQN, an explainable heuristic, and a random baseline')
        self.p('BIL353 - Artificial Intelligence Term Project')
        self.p('Software Engineering | Istanbul Ticaret University')
        self.story.append(Spacer(1, 17 * mm))
        self.table(['DELIVERED STUDY', 'RECORDED CONFIGURATION'], [
            ['Environment', '50 x 50 cells | 3 agents | 375 green + 375 red'],
            ['Training', f'{runs} independent repeats x {episodes} episodes'],
            ['Held-out evaluation', f'{maps} maps x {runs} repeats x 4 policy conditions'],
            ['Episode horizon', f'{horizon} decision steps in the delivered experiment'],
            ['Reward', '+1 green / -1 red / no reward shaping'],
        ], [175, 305])
        self.p('Report generated from recorded CSV and JSON results. The supplied project brief, '
               'rather than an independently supplied marking rubric, defines the requirements.')
        self.p('Implementation and experiment version: 17 October 2026. Submission deadline: 26 November 2026.')
        self.page()

        self.heading('1. Abstract')
        self.p(f'This project implements a reproducible three-agent foraging environment on a '
               f'50 x 50 grid and compares random action selection, a risk-aware heuristic and '
               f'independent Double Deep Q-Networks. Exactly 375 green and 375 red food cells '
               f'are placed initially, and each agent receives only its own state, visible food '
               f'and the location of its nearest neighbor. The original +1/-1 food rewards are '
               f'preserved. {runs} independent network triplets were trained for {episodes} episodes '
               f'each; evaluation used {maps} held-out maps and a {horizon}-step finite horizon. '
               f'Mean homogeneous team rewards were {random_reward:.2f} for Random, {hr:.2f} for '
               f'Heuristic and {rr:.2f} for Double DQN. Results are accompanied by run/map bootstrap '
               f'intervals, raw episode logs and explicit limitations. These findings establish '
               f'performance under the tested conditions, not optimality or universal superiority.')
        self.heading('2. Introduction')
        self.p('Resource foraging provides a compact setting for studying sequential decisions, '
               'negative outcomes and competition. A useful solution must connect a precise '
               'transition model to credible experiments: visually plausible movement alone '
               'does not establish intelligent behavior or learning.')
        self.p('The objectives are to implement the assignment constraints, compare three '
               'decision mechanisms under matched conditions, test whether experience improves '
               'a learned policy, and deliver a readable simulation with reproducible evidence. '
               'The learning objective follows the interaction-and-reward framework of Sutton '
               'and Barto [1]. No particular grade or empirical outcome is assumed.')
        self.heading('3. Problem Definition')
        self.p('Let the board be B in {-1,0,+1}^(50x50), with -1 red, 0 empty and +1 green. '
               'Three agents choose one action from {Up, Down, Left, Right, No Action} at each '
               'decision. A speed independently sampled from {1,2,3} determines the maximum '
               'straight-line displacement. Food is consumed at visited intermediate cells.')
        self.formula('r_i,t = green_i,t - red_i,t\nR_i = sum_t r_i,t     ;     R_team = sum_i R_i')
        self.p('Each agent maximizes its own expected discounted return. Team reward is an '
               'evaluation and model-selection statistic, not a shared training reward. '
               'Because a third agent is hidden from each individual observation, the task is '
               'partially observable from an agent perspective.')
        self.page()

        self.heading('4. Related Work')
        self.p('Sutton and Barto [1] formalize action values, temporal-difference learning and '
               'exploration. Tabular action-value methods provide the conceptual starting '
               'point, but enumerating arbitrary food configurations on this board is infeasible '
               'without substantial abstraction.')
        self.p('Mnih et al. [2] demonstrated deep Q-learning using experience replay and a target '
               'network. This project adapts those mechanisms to a small action-conditioned '
               'network with engineered vector features; it does not reproduce their visual '
               'Atari architecture or benchmark. Van Hasselt et al. [3] separate next-action '
               'selection from evaluation to reduce overestimation. This motivates the Double '
               'DQN target used here, without guaranteeing improvement in this environment.')
        self.p('Tan [4] distinguishes independent and cooperative multi-agent learners. The '
               'present agents have separate parameters, replay memories and rewards and '
               'do not communicate. Henderson et al. [5] discuss variation and reproducibility '
               'in deep RL, motivating multiple training repeats and uncertainty reporting.')
        self.heading('5. Methodology')
        self.p('The implementation proceeds from a tested environment contract to baselines, '
               'learning, held-out evaluation and visualization. All policies accept an '
               'observation vector; they cannot access the Environment object. Headless '
               'training and evaluation do not depend on Pygame.')
        self.p('Random samples all five actions uniformly. Heuristic selects a green using '
               'distance plus a penalty when the observed neighbor is closer. It scores '
               'candidate moves using immediate green count, red risk, a reverse-Dijkstra '
               'cost-to-go to the selected target, and a small self-history oscillation term. '
               'A wait cost is part of this decision score, not an environment penalty. These scores are decision rules and '
               'never alter the environment reward. Double DQN chooses the action with '
               'the greatest learned value, except during epsilon-greedy training exploration.')
        self.formula('Dijkstra edge cost(u -> v) = 1 + 6 * red(v)\n'
                     'Heuristic score(a) = 3 G_path(a) - 6 C_path(a)\n'
                     '                     - path_length(a) - cost_to_go(endpoint(a))\n'
                     '                     - small oscillation / waiting terms')
        self.p('The principal comparison uses homogeneous populations (three policies of '
               'one algorithm). An additional mixed population contains one of each policy, '
               'with starting slots rotated across maps. This separates homogeneous resource '
               'collection from direct competition between unlike strategies.')
        self.p('An initial one-step heuristic sometimes waited at red barriers. This was '
               'diagnosed on an exploratory map set (900000..900059), and the heuristic '
               'was revised to use risk-aware path planning. That exploratory summary is '
               'archived separately. The final study below uses a fresh map set; the learned '
               'networks were not retuned using those final scores.')
        self.page()

        self.heading('6. Environment and Agent Design')
        self.table(['COMPONENT', 'RESPONSIBILITY'], [
            ['Environment', 'Reset, fixed rules, episode horizon and common transition'],
            ['MovementResolver', 'Synchronous substeps, bounds and food arbitration'],
            ['Agent / Food', 'Individual position and counts; board food encoding'],
            ['RewardManager', 'Unshaped individual +1/-1 consumption reward'],
            ['ObservationManager', 'Food visibility and nearest-neighbor information filter'],
            ['TrainingManager', 'Separate learners, validation and model selection'],
            ['ExperimentEvaluator', 'Paired maps, raw logs and uncertainty estimation'],
            ['SimulationRenderer', 'Dashboard, animation, controls and spectator view'],
        ], [138, 342])
        self.p('The 2508-element float32 observation contains 2500 food entries followed by '
               'normalized self coordinates (2), nearest-neighbor coordinates (2), own '
               'speed (1), distance-metric indicators (2), and remaining horizon (1). '
               'The food board contains no agent occupancy. Neighbor identity, speed, '
               'targets and scores are not provided.')
        self.p('At reset, Manhattan or Euclidean distance is chosen with equal probability '
               'and held fixed. Manhattan is |dr|+|dc| and Euclidean is sqrt(dr^2+dc^2). '
               'Ties use a seeded choice. Initial agent cells are distinct and food-free. '
               'Independent random namespaces control layouts, metrics, speeds and arbitration.')
        self.p('Ambiguous-rule assumptions: speed is resampled before every decision; all '
               'actions are simultaneous; movement is resolved cell by cell; borders stop '
               'movement without penalty; agents may overlap or swap; food never respawns. '
               'A green contested in the same substep goes to a Left mover if available. '
               'Any remaining tie, or a red contest, uses seeded random arbitration. Earlier '
               'arrival beats later arrival. Food is consumed exactly once. All food depleted '
               'or the finite horizon ends the episode. These assumptions require comparison '
               'with any later instructor clarification; they are not attributed to the brief.')
        if tests is not None:
            self.p(f'The delivered verification record contains {tests} automated tests. '
                   'Tests cover geometry, density, reward accounting, all actions, bounds, '
                   'simultaneous competition, hidden-agent invariance, speed support, '
                   'repeatability, conservation, learning updates, saved-model equivalence '
                   'and renderer controls. Test success is evidence for tested scenarios, '
                   'not a proof that every possible defect is absent.')
        self.page()

        self.heading('7. Reinforcement Learning Algorithm')
        self.p('Each learner evaluates Q_theta(o,a) = MLP_theta(phi(o,a)). The visible-information '
               'encoder produces 22 features for each of the five candidate actions. Six '
               'indicators describe green/red presence in each of three path cells. Other '
               'features describe the nearest remaining green relative to the endpoint, '
               'observed-neighbor geometry, movement length, own speed, metric, action '
               'identity, remaining green fraction and remaining time. The neighbor-to-target '
               'advantage feature uses Manhattan geometry as an approximation even when '
               'neighbor selection uses Euclidean distance.')
        self.p('The network is 22-32-1 with tanh hidden activations (769 parameters per agent). '
               'A single network scores all action descriptions. It receives no demonstration '
               'actions, no heuristic override and no shaped reward. Feature engineering '
               'provides a strong inductive bias and loses some full-map information; '
               'the method is not end-to-end learning from raw pixels.')
        self.formula('a* = argmax_b Q_online(o_next, b)\n'
                     'y = r + gamma (1 - done) Q_target(o_next, a*)\n'
                     'error = Q_online(o, a) - y\n'
                     'Huber(error) = 0.5 error^2        if |error| <= 1\n'
                     '               |error| - 0.5     otherwise')
        self.p('A replay minibatch minimizes Huber loss with Adam. Gradients are globally '
               'clipped at norm 10. Every 200 gradient updates the target weights are '
               'copied from the online network. At the finite task horizon done=1; '
               'the learner does not bootstrap beyond the episode budget. Remaining time '
               'is observable. Partial observability and co-learning opponents prevent '
               'a general convergence or optimality claim.')
        self.table(['HYPERPARAMETER', 'VALUE'], [
            ['Discount / learning rate', '0.90 / 0.001'],
            ['Replay / warm-up / batch', '20000 per agent / 256 transitions / 64'],
            ['Update frequency / target sync', 'Every 4 transitions / 200 gradient updates'],
            ['Optimizer / loss', 'Adam (0.9, 0.999, epsilon=1e-8) / Huber delta=1'],
            ['Exploration', 'epsilon=max(0.05, 1-0.95*episode/(0.8*episodes))'],
            ['Inference / checkpoints', 'epsilon=0; NPZ weights and encoder version'],
        ], [190, 290])
        self.p('NPZ models are inference checkpoints. Optimizer, replay and generator states '
               'are not saved for exact mid-training resumption. last_agent files preserve '
               'final weights separately from validation-selected weights.')
        self.page()

        self.heading('8. Experimental Setup')
        self.p(f'The delivered experiment used {runs} independent training repeats, each with '
               f'three simultaneously learning agents for {episodes} episodes of up to '
               f'{horizon} decisions. Every ten episodes and at episode zero, greedy validation '
               'ran on three fixed maps. The best validation team score selected a network '
               'triplet within each run. No final-test scores were used to select checkpoints.')
        self.table(['ITEM', 'RECORDED SETTING'], [
            ['Training maps', '10000 + 100000*run + zero-based episode index'],
            ['Validation maps', '500000..500002 (three fixed maps, reused)'],
            ['Final test maps', f"{s['test_seeds'][0]}..{s['test_seeds'][1]}"],
            ['Conditions', 'Homogeneous Random, Heuristic, RL; one-of-each mixed'],
            ['Evaluation budget', f'{runs * maps * 4} episodes; {runs * maps * 4 * 3} agent records'],
            ['Food / metric / speed', '375+375; equal metric probability; uniform speed 1..3'],
            ['Primary outcome', 'Mean homogeneous team episode reward'],
            ['Uncertainty', '2000 crossed run/map bootstrap resamples, percentile 95% CI'],
        ], [160, 320])
        self.p('Policies use independent random generators. Layout and per-step speeds are '
               'keyed independently of policy randomness, so algorithms share the same '
               'exogenous scenario for a given map. Food ties use event-keyed randomness; '
               'different behavior can still create different contests. The mixed condition '
               'rotates the three policy types across slots in a three-map cycle.')
        self.p('Individual reward, green/red counts, green collection fraction, steps and '
               'reward per step are logged. Team green fraction is total green divided by '
               '375. Overlap counts are pair-substep occupancy events, including stationary '
               'overlap; swaps are pair-substep exchanges; competitions are food-cell contests. '
               'These environment events appear identically on the three agent rows and '
               'must not be summed across those rows.')
        self.p('For uncertainty, the reward matrix has independent run and paired-map axes; '
               'both axes are resampled. Agents are aggregated before inference. Paired '
               'RL-minus-baseline intervals use differences on the same run/map coordinates. '
               'Only five runs limit precision. Evaluation is on new maps from the same '
               'generator distribution, not an out-of-distribution generalization test.')
        self.p(f'Execution uses Python {platform.python_version()} with NumPy and CPU computation. '
               'The exact environment versions are recorded in requirements-lock.txt and '
               'runtime.json. Interactive default horizon is 500, whereas the recorded '
               'models and experiments use 300; their results must not be conflated.')
        self.page()

        self.heading('9. Results and Evaluation')
        result_rows = []
        for name, label in [('random', 'Random'), ('heuristic', 'Heuristic'), ('rl', 'Double DQN')]:
            entry = algorithms[name]
            lo, hi = entry['team_reward_ci95']
            result_rows.append([label, f"{entry['mean_team_reward']:.2f}",
                                f"{entry['std_team_reward']:.2f}", f'[{lo:.2f}, {hi:.2f}]'])
        self.table(['ALGORITHM', 'TEAM MEAN', 'TEAM SD', '95% CI'], result_rows, [130, 100, 100, 150])
        diff = s['paired_differences']
        for key, label in [('rl_minus_random', 'Random'), ('rl_minus_heuristic', 'Heuristic')]:
            item = diff[key]
            lo, hi = item['ci95']
            self.p(f"The paired mean Double-DQN-minus-{label} team reward difference was "
                   f"{item['mean_team_reward_difference']:.2f}, with 95% bootstrap interval "
                   f'[{lo:.2f}, {hi:.2f}]. This comparison applies to the fixed tested budget.')
        self.figure('algorithm_comparison', 'Figure 1. Homogeneous held-out team reward. Error bars are crossed-bootstrap 95% intervals.')
        self.p('Team SD describes the spread of run/map episode outcomes, while the CI '
               'describes uncertainty in the estimated mean. These are different quantities. '
               'The experiment does not establish performance on a different food density '
               'or on a longer episode horizon.')
        self.page()

        self.heading('Results: food and individual agents')
        self.figure('food_collection', 'Figure 2. Mean green and red food consumed per individual agent.')
        self.figure('agent_comparison', 'Figure 3. Mean reward by starting agent slot in homogeneous populations.')
        food_rows = []
        for name in ['random', 'heuristic', 'rl']:
            item = algorithms[name]
            m = item['per_agent_metrics']
            food_rows.append([name, f"{m['green']['mean']:.2f}", f"{m['red']['mean']:.2f}",
                              f"{100 * item['mean_team_green_rate']:.1f}%",
                              f"{m['reward_per_step']['mean']:.4f}"])
        self.table(['POLICY', 'GREEN/AGENT', 'RED/AGENT', 'TEAM GREEN %', 'REWARD/STEP'], food_rows,
                   [80, 100, 100, 100, 100])
        self.page()

        self.heading('Results: learning and unseen maps')
        self.figure('learning_curve', 'Figure 4. Training reward with +/-1 SD band, and greedy validation by independent run.')
        self.figure('episode_reward', 'Figure 5. Team reward on each held-out map, averaged over independent runs.')
        validation_rows = []
        for run in self.runs:
            with (run / 'training.csv').open() as stream:
                history = list(csv.DictReader(stream))
            selection = json.loads((run / 'selection.json').read_text())
            validation_rows.append([run.name, f"{float(history[0]['validation_reward']):.2f}", selection['episode'],
                                    f"{selection['validation_team_reward']:.2f}"])
        self.table(['RUN', 'UNTRAINED VALIDATION', 'SELECTED EPISODE', 'SELECTED VALIDATION'],
                   validation_rows, [70, 150, 130, 130])
        self.page()

        self.heading('Results: mixed competition and interactions')
        self.figure('mixed_competition', 'Figure 6. Individual reward when one of each policy shares the same environment.')
        mixed_rows = []
        for name in ['random', 'heuristic', 'rl']:
            subset = [r for r in self.rows if r['mode'] == 'mixed' and r['policy'] == name]
            mixed_rows.append([name, f"{np.mean([float(r['reward']) for r in subset]):.2f}",
                               f"{np.mean([float(r['green']) for r in subset]):.2f}",
                               f"{np.mean([float(r['red']) for r in subset]):.2f}"])
        self.table(['MIXED POLICY', 'REWARD', 'GREEN', 'RED'], mixed_rows)
        event_rows = []
        for name in ['random', 'heuristic', 'rl']:
            m = algorithms[name].get('environment_metrics', algorithms[name]['per_agent_metrics'])
            individual = algorithms[name]['per_agent_metrics']
            event_rows.append([name, f"{m['competitions']['mean']:.3f}",
                               f"{m['left_priority_wins']['mean']:.3f}",
                               f"{m['overlaps']['mean']:.2f}", f"{individual['steps']['mean']:.1f}"])
        self.table(['MODE', 'CONTESTS', 'LEFT WINS', 'OVERLAPS', 'STEPS'], event_rows)
        self.p('Interaction counts above are per environment episode, not per agent. '
               'Naturally occurring food contests may be rare in sparse spatial encounters; '
               'controlled tests therefore provide the direct evidence that Left priority '
               'is implemented. Mixed-policy results describe a different opponent population '
               'from the homogeneous training population.')
        self.page()

        self.heading('10. Discussion')
        relation = 'higher' if rr > hr else 'lower'
        self.p(f'Double DQN achieved a {relation} homogeneous mean team reward than the '
               f'heuristic in this experiment ({rr:.2f} versus {hr:.2f}). The paired interval '
               'above, rather than the means alone, should guide interpretation of that '
               'difference. Validation trajectories show learning variation and occasional '
               'regression, which is why checkpoint selection is separated from final testing.')
        self.p('The heuristic is competitive because food is fully visible and short straight '
               'paths can be evaluated directly. RL benefits from a feature encoder that '
               'describes those same observable path opportunities. Its performance is '
               'therefore evidence about learning an action-value mapping under engineered '
               'features, not evidence that a network independently discovered every useful '
               'spatial abstraction. The separate Random baseline helps expose the value '
               'of exploiting visible structure.')
        self.heading('11. Limitations')
        self.p('The study uses five training repeats, three repeatedly reused validation '
               'maps, one food density and a 300-step evaluation horizon. Hyperparameters '
               'were fixed without an extensive search. New maps share the training generator; '
               'robustness to changed layouts, respawn, blocking collisions or alternate '
               'speed timing is unmeasured. Partial observations and a lossy feature encoder '
               'create state aliasing, and independent co-learning can be nonstationary.')
        self.p('The aggregate heuristic-to-RL comparison does not isolate Double DQN versus '
               'ordinary DQN, or the effect of each feature; such ablations remain future '
               'experiments. Model files do not preserve complete training state. GUI tests '
               'exercise headless rendering and controls, not every desktop configuration. '
               'Ambiguous assignment rules are documented assumptions and should be checked '
               'against instructor clarification before submission.')
        self.heading('12. Conclusion')
        self.p('The project delivers a tested multi-agent environment, three runnable policy '
               'families, independent learned models, a Pygame dashboard, matched held-out '
               'experiments and a report tied to raw data. Its contribution is an inspectable '
               'connection between assignment rules, algorithm behavior and empirical '
               'evidence. Further work should expand validation maps, training repeats and '
               'feature/algorithm ablations before claiming broader superiority.')
        self.page()

        self.heading('13. References')
        for index, (description, url) in enumerate(REFERENCES, 1):
            self.p(f'[{index}] {description}')
            self.story.append(Paragraph(f'<link href="{html.escape(url)}" color="#087c91">'
                                        f'{html.escape(url)}</link>', self.styles['CaptionCustom']))
            self.markdown.extend([url, ''])
        self.story.append(Spacer(1, 15 * mm))
        self.p('Reproduction: README.md documents installation, test, training, evaluation '
               'and report commands. docs/ASSUMPTIONS.md is the transition contract. '
               'results/episodes.csv contains the full evaluation records; results/summary.json '
               'contains aggregates and intervals; models/run_*/training.csv contains '
               'learning histories. requirements-lock.txt records exact package versions. '
               'MANIFEST.json provides SHA256 integrity checks.')
        self.p('The supplied brief is a project requirement source, not a scientific '
               'publication. No external paper is used to imply that the measured project '
               'results reproduce the paper benchmarks.')

        self.output.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(str(self.output), pagesize=(210 * mm, 297 * mm),
                                rightMargin=20 * mm, leftMargin=20 * mm,
                                topMargin=21 * mm, bottomMargin=20 * mm,
                                title='BIL353 - Multi-Agent Food Foraging',
                                author='BIL353 Project', pageCompression=1)

        def decorate(canvas, document):
            canvas.saveState()
            canvas.setStrokeColor(colors.HexColor('#c7dbe4'))
            canvas.line(20 * mm, 282 * mm, 190 * mm, 282 * mm)
            canvas.setFont('Helvetica', 8)
            canvas.setFillColor(colors.HexColor('#62718a'))
            canvas.drawString(20 * mm, 286 * mm, 'BIL353  /  MULTI-AGENT FOOD FORAGING')
            canvas.drawString(20 * mm, 12 * mm, 'Recorded experiments | Independent Double DQN')
            canvas.drawRightString(190 * mm, 12 * mm, str(document.page))
            canvas.restoreState()

        doc.build(self.story, onFirstPage=decorate, onLaterPages=decorate)
        (Path('report') / (self.output.stem + '.md')).write_text('\n'.join(self.markdown), encoding='utf-8')
        short = ['# Recorded experimental results', '',
                 f'{runs} independent repeats, {maps} unseen maps, {horizon} steps, no shaping.', '',
                 '| Policy | Mean team reward | 95% CI |', '|---|---:|---|']
        for row in result_rows:
            short.append(f'| {row[0]} | {row[1]} | {row[3]} |')
        short.extend(['', 'These are measured results; see summary.json for paired differences.',
                      'Training and final test seeds are disjoint. The 500-step GUI default is a different horizon.'])
        (self.results / 'RESULTS.md').write_text('\n'.join(short), encoding='utf-8')
        print(f'Built {self.output} from recorded results.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--models', type=Path, default=Path('models'))
    parser.add_argument('--output', type=Path, default=Path('Project_Report.pdf'))
    args = parser.parse_args()
    ReportBuilder(args.results, args.models, args.output).build()


if __name__ == '__main__':
    main()
