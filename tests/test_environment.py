import numpy as np
import pytest
from agent import Action
from config import Config
from environment import Environment
from observations import decode, distance


def scene(positions=((10, 10), (30, 30), (40, 40)), speeds=(1, 1, 1)):
    env = Environment()
    env.board[:] = 0
    # Keep episodes alive during isolated movement tests.
    env.board[49, 49] = -1
    for agent, pos, speed in zip(env.agents, positions, speeds):
        agent.position, agent.speed = pos, speed
    return env


@pytest.mark.parametrize('seed', range(10))
def test_initial_invariants(seed):
    env = Environment()
    observations = env.reset(seed)
    assert env.board.shape == (50, 50)
    assert np.count_nonzero(env.board == 1) == np.count_nonzero(env.board == -1) == 375
    assert np.count_nonzero(env.board) <= 750
    assert len(env.agents) == 3
    assert len({a.position for a in env.agents}) == 3
    for agent, observation in zip(env.agents, observations):
        assert all(0 <= c < 50 for c in agent.position)
        assert env.board[agent.position] == 0
        assert 1 <= agent.speed <= 3
        assert observation.shape == (2508,)
        assert observation.dtype == np.float32
        assert np.array_equal(observation[:2500].reshape(50, 50), env.board)


@pytest.mark.parametrize('action,expected', [(0, (7, 10)), (1, (13, 10)),
                                         (2, (10, 7)), (3, (10, 13)), (4, (10, 10))])
def test_all_five_actions(action, expected):
    env = scene(speeds=(3, 1, 1))
    env.step([action, 4, 4])
    assert env.agents[0].position == expected


def test_path_food_and_independent_rewards():
    env = scene(speeds=(3, 1, 1))
    env.board[10, 11], env.board[10, 12], env.board[10, 13] = 1, -1, 1
    _, reward, _, _ = env.step([3, 4, 4])
    assert reward.tolist() == [1, 0, 0]
    assert (env.agents[0].green, env.agents[0].red) == (2, 1)
    assert not np.any(env.board[10, 11:14])
    env.agents[0].speed = 3
    _, reward, _, _ = env.step([2, 4, 4])
    assert reward.sum() == 0


@pytest.mark.parametrize('action,pos', [(0, (0, 10)), (1, (49, 10)),
                                      (2, (10, 0)), (3, (10, 49))])
def test_boundary_stop(action, pos):
    env = scene(positions=(pos, (30, 30), (40, 40)), speeds=(3, 1, 1))
    _, rewards, _, info = env.step([action, 4, 4])
    assert env.agents[0].position == pos
    assert rewards.sum() == 0
    assert info['events']['boundary_hits'] == 1


@pytest.mark.parametrize('left_identifier', [0, 1, 2])
def test_left_priority_independent_of_agent_order(left_identifier):
    env = scene()
    positions = [(10, 9), (9, 10), (11, 10)]
    positions[left_identifier] = (10, 11)
    actions = [3, 1, 0]
    actions[left_identifier] = 2
    for agent, position in zip(env.agents, positions):
        agent.position = position
    env.board[10, 10] = 1
    _, rewards, _, info = env.step(actions)
    assert rewards[left_identifier] == 1
    assert rewards.sum() == 1
    assert info['events']['left_priority_wins'] == 1
    assert env.board[10, 10] == 0


def test_earlier_arrival_beats_later_left():
    env = scene(positions=((10, 9), (10, 12), (40, 40)), speeds=(1, 2, 1))
    env.board[10, 10] = 1
    _, rewards, _, _ = env.step([3, 2, 4])
    assert rewards.tolist() == [1, 0, 0]


def test_red_competition_single_consumer():
    env = scene(positions=((10, 9), (10, 11), (40, 40)))
    env.board[10, 10] = -1
    _, rewards, _, info = env.step([3, 2, 4])
    assert rewards.sum() == -1
    assert sum(a.red for a in env.agents) == 1
    assert info['events']['competitions'] == 1


def test_nearest_only_and_hidden_agent_invariance():
    env = scene(positions=((10, 10), (10, 11), (40, 40)))
    before = env.observe()[0]
    env.agents[2].position = (45, 45)
    after = env.observe()[0]
    assert np.array_equal(before, after)
    assert np.array_equal(decode(after)[2], [10, 11])
    env.agents[1].speed = 3
    env.agents[1].green = 20
    assert np.array_equal(after, env.observe()[0])


def test_metric_can_change_nearest_neighbor():
    env = scene(positions=((0, 0), (0, 5), (3, 3)))
    env.metric = 'manhattan'
    assert np.array_equal(decode(env.observe()[0])[2], [0, 5])
    env.metric = 'euclidean'
    assert np.array_equal(decode(env.observe()[0])[2], [3, 3])
    assert distance((0, 0), (3, 4), 'manhattan') == 7
    assert distance((0, 0), (3, 4), 'euclidean') == 5


def test_ties_are_reproducible_and_observation_has_no_rng_side_effects():
    env = scene(positions=((10, 10), (10, 9), (10, 11)))
    assert np.array_equal(env.observe()[0], env.observe()[0])


def test_seed_reproducibility_and_exogenous_speeds():
    a, b = Environment(), Environment()
    a.reset(2026)
    b.reset(2026)
    assert a.metric == b.metric
    for _ in range(60):
        actions = [0, 2, 3]
        oa, ra, da, ia = a.step(actions)
        ob, rb, db, ib = b.step(actions)
        assert all(np.array_equal(x, y) for x, y in zip(oa, ob))
        assert np.array_equal(ra, rb)
        assert ia == ib and da == db
    a.reset(100)
    b.reset(100)
    for _ in range(30):
        a.step([0, 0, 0])
        b.step([4, 4, 4])
        assert [p.speed for p in a.agents] == [p.speed for p in b.agents]


def test_speed_support_and_metric_randomization():
    env = Environment()
    seen, metrics = set(), set()
    for seed in range(100):
        env.reset(seed)
        seen.update(a.speed for a in env.agents)
        metrics.add(env.metric)
    assert seen == {1, 2, 3}
    assert metrics == {'manhattan', 'euclidean'}


def test_conservation_over_random_episode():
    env = Environment(Config(max_steps=100))
    rng = np.random.default_rng(33)
    total = np.zeros(3)
    while not env.done:
        _, reward, _, _ = env.step(list(rng.integers(5, size=3)))
        total += reward
        assert np.count_nonzero(env.board == 1) + sum(a.green for a in env.agents) == 375
        assert np.count_nonzero(env.board == -1) + sum(a.red for a in env.agents) == 375
        assert all(0 <= coordinate < 50 for a in env.agents for coordinate in a.position)
    assert total.tolist() == [a.score for a in env.agents]


def test_horizon_and_done_guard():
    env = Environment(Config(max_steps=1))
    _, _, done, info = env.step([4, 4, 4])
    assert done and info['truncated'] and not info['terminated']
    with pytest.raises(RuntimeError):
        env.step([4, 4, 4])


def test_food_exhaustion_terminates():
    env = scene()
    env.board[:] = 0
    env.board[10, 11] = 1
    _, _, done, info = env.step([3, 4, 4])
    assert done and info['terminated'] and not info['truncated']


@pytest.mark.parametrize('actions', [[0], [0, 1, 5], [True, 1, 2], [0., 1, 2]])
def test_invalid_actions(actions):
    with pytest.raises(ValueError):
        Environment().step(actions)


@pytest.mark.parametrize('kwargs', [{'size': 49}, {'agents': 4}, {'food_per_color': 376},
                                   {'food_per_color': 0}, {'max_steps': 0},
                                   {'food_per_color': 2.5}, {'max_steps': True}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)
