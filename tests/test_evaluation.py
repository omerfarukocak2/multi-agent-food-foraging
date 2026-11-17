import numpy as np
import json
import pytest
from evaluation.evaluate import bootstrap_interval, ExperimentEvaluator


def test_bootstrap_constant_and_reproducibility():
    values = np.full((5, 60), 7.)
    assert bootstrap_interval(values) == [7., 7.]
    varied = np.arange(100).reshape(5, 20)
    a = bootstrap_interval(varied)
    assert a == bootstrap_interval(varied)
    assert a[0] < varied.mean() < a[1]


def test_evaluator_rejects_training_map_leakage(tmp_path):
    model_root = tmp_path / 'models'
    run = model_root / 'run_0'
    run.mkdir(parents=True)
    (run / 'agent_0.npz').touch()
    (run / 'training_config.json').write_text(json.dumps({
        'training_seed_start': 10000, 'episodes': 10,
        'validation_seed_start': 500000, 'validation_maps': 3}))
    evaluator = ExperimentEvaluator(model_root, tmp_path / 'results', maps=3,
                                    steps=300, seed_start=10000)
    with pytest.raises(ValueError, match='overlap'):
        evaluator.evaluate()
