import numpy as np
from algorithms.rl_agent import RLAgent, Network
from algorithms.features import action_features
from algorithms.heuristic_agent import HeuristicAgent
from environment import Environment


def test_model_roundtrip(tmp_path):
    obs = Environment().observe()[0]
    agent = RLAgent(77)
    q = agent.online.predict(action_features(obs))
    agent.save(tmp_path / 'test.npz')
    restored = RLAgent.load(tmp_path / 'test.npz')
    assert np.array_equal(q, restored.online.predict(action_features(obs)))
    assert agent.act(obs) == restored.act(obs)


def test_supervised_gradient_step_reduces_huber_loss():
    rng = np.random.default_rng(4)
    net = Network(rng)
    x = rng.normal(size=(100, 22)).astype(np.float32)
    target = .4 * x[:, 0] - .3 * x[:, 1]
    before = ((net.predict(x) - target) ** 2).mean()
    for _ in range(100):
        loss = net.fit(x, target, .003)
    after = ((net.predict(x) - target) ** 2).mean()
    assert after < before * .1
    assert np.isfinite(loss)


def test_replay_learns_terminal_reward_without_shaping():
    agent = RLAgent(9)
    f = np.zeros((5, 22), np.float32)
    f[0, 0] = 1
    for _ in range(600):
        agent.remember(f, 0, 1, f, True)
        agent.learn()
    assert agent.online.predict(f)[0] > .7


def test_double_dqn_selects_online_and_evaluates_target():
    agent = RLAgent(7)
    agent.count, agent.transitions = 256, 4
    agent.x[:256] = 0
    agent.next_x[:256] = 0
    agent.rewards[:256] = 1
    agent.dones[:256] = 0
    class Online:
        updates = 1
        def predict(self, x):
            return np.tile([1., 7., 2., 3., 0.], (len(x), 1))
        def fit(self, x, targets, learning_rate):
            # Online picks action 1; target action 1 has value 2, not max 99.
            assert np.allclose(targets, 1 + .9 * 2)
            return .25
    class Target:
        def predict(self, x):
            return np.tile([99., 2., 3., 4., 5.], (len(x), 1))
    agent.online, agent.target_net = Online(), Target()
    assert agent.learn() == .25


def test_encoder_does_not_duplicate_boundary_food():
    env = Environment()
    env.board[:] = 0
    env.agents[0].position = (0, 1)
    env.agents[0].speed = 3
    env.board[0, 0] = 1
    f = action_features(env.observe()[0])
    assert f[2, [0, 2, 4]].sum() == 1
    assert np.isfinite(f).all()


def test_heuristic_collects_adjacent_green_and_avoids_red():
    env = Environment()
    env.board[:] = 0
    env.agents[0].position = (10, 10)
    env.agents[0].speed = 1
    env.board[10, 11] = 1
    env.board[10, 9] = -1
    assert HeuristicAgent().act(env.observe()[0]) == 3


def test_heuristic_routes_around_red_wall_instead_of_waiting():
    env = Environment()
    env.board[:] = 0
    env.agents[0].position = (10, 10)
    env.agents[1].position = (40, 40)
    env.agents[2].position = (45, 45)
    env.board[10, 12] = 1
    env.board[9:12, 11] = -1
    policy = HeuristicAgent()
    for _ in range(20):
        env.agents[0].speed = 1
        env.step([policy.act(env.observe()[0]), 4, 4])
        if env.agents[0].green:
            break
    assert env.agents[0].green == 1
    assert env.agents[0].red == 0
