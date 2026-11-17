"""Verify recorded evidence, render the report, and record the execution environment."""
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import xml.etree.ElementTree as ET
import numpy as np
from pypdf import PdfReader
from algorithms.rl_agent import RLAgent
from algorithms.features import action_features
from environment import Environment


def main():
    results = Path('results')
    summary_path = results / 'summary.json'
    summary = json.loads(summary_path.read_text())
    # Migrate pre-release grouping only; all numeric measurements remain unchanged.
    individual_keys = ['reward', 'green', 'red', 'green_collection_rate', 'steps', 'reward_per_step']
    for entry in summary['algorithms'].values():
        metrics = entry['per_agent_metrics']
        entry.setdefault('environment_metrics', {k: v for k, v in metrics.items() if k not in individual_keys})
        entry['per_agent_metrics'] = {k: v for k, v in metrics.items() if k in individual_keys}
    summary_path.write_text(json.dumps(summary, indent=2))
    with (results / 'episodes.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    expected = summary['maps'] * summary['runs'] * 4 * 3
    assert len(rows) == expected
    assert all(int(r['reward']) == int(r['green']) - int(r['red']) for r in rows)
    assert all(0 < int(r['steps']) <= summary['config']['max_steps'] for r in rows)
    groups = {}
    for row in rows:
        groups.setdefault((row['mode'], row['run'], row['seed']), []).append(row)
    for group in groups.values():
        assert len(group) == 3
        assert sum(int(r['green']) for r in group) <= 375
        assert sum(int(r['red']) for r in group) <= 375
    for name in ['random', 'heuristic', 'rl']:
        rewards = [sum(int(r['reward']) for r in group) for key, group in groups.items() if key[0] == name]
        assert np.isclose(np.mean(rewards), summary['algorithms'][name]['mean_team_reward'])
    env = Environment()
    observation = env.reset(950099)[0]
    models = sorted(Path('models').glob('run_*/agent_*.npz'))
    assert len(models) == summary['runs'] * 3
    for path in models:
        policy = RLAgent.load(path)
        values = policy.online.predict(action_features(observation))
        assert values.shape == (5,) and np.isfinite(values).all()
    root = ET.parse(results / 'test_results.xml').getroot()
    suites = root.findall('testsuite') if root.tag != 'testsuite' else [root]
    tests = sum(int(s.attrib.get('tests', 0)) for s in suites)
    failures = sum(int(s.attrib.get('failures', 0)) + int(s.attrib.get('errors', 0)) for s in suites)
    assert failures == 0
    runtime = {'python': platform.python_version(), 'platform': platform.platform(),
               'packages': {m: importlib.metadata.version(m) for m in
                            ['numpy', 'pygame', 'matplotlib', 'pytest', 'reportlab', 'pypdf', 'pymupdf']}}
    (results / 'runtime.json').write_text(json.dumps(runtime, indent=2))
    reader = PdfReader('output/pdf/Project_Report_Revised.pdf')
    text = '\n'.join(p.extract_text() for p in reader.pages)
    headings = ['Abstract', 'Introduction', 'Problem Definition', 'Related Work', 'Methodology',
                'Environment and Agent Design', 'Reinforcement Learning Algorithm',
                'Experimental Setup', 'Results and Evaluation', 'Discussion', 'Limitations',
                'Conclusion', 'References']
    assert all(h in text for h in headings)
    for name in ['algorithm_comparison', 'food_collection', 'agent_comparison',
                 'episode_reward', 'learning_curve', 'mixed_competition']:
        assert (results / f'{name}.png').exists() and (results / f'{name}.pdf').exists()
    verification = {'tests_passed': tests, 'test_failures_or_errors': failures,
                    'evaluation_episodes': len(groups), 'agent_records': len(rows),
                    'selected_models_loaded': len(models), 'pdf_pages': len(reader.pages),
                    'all_13_report_sections_present': True,
                    'recorded_rewards_and_food_conservation_checked': True,
                    'aggregate_rewards_recomputed': True}
    (results / 'verification.json').write_text(json.dumps(verification, indent=2))
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
