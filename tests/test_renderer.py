import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
import pygame
from visualization.renderer import SimulationRenderer


def test_controls_reset_and_renderer_does_not_change_state(tmp_path):
    view = SimulationRenderer(seed=12, mode='heuristic')
    view.draw()
    view.control('play')
    assert view.running
    view.advance()
    assert view.env.steps == 1
    score = [a.score for a in view.env.agents]
    positions = [a.position for a in view.env.agents]
    view.draw(.2)
    view.draw(.8)
    assert score == [a.score for a in view.env.agents]
    assert positions == [a.position for a in view.env.agents]
    view.control('reset')
    assert view.env.steps == 0 and view.seed == 12
    view.control('new')
    assert view.seed == 13
    view.control('mode_random')
    assert all(p.name == 'Random' for p in view.policies)
    old = view.trails
    view.control('trails')
    assert view.trails != old
    view.control('faster')
    assert view.speed == 16
    view.run(frames=1, snapshot=tmp_path / 'dashboard.png')
    assert (tmp_path / 'dashboard.png').stat().st_size > 10000
    assert not pygame.get_init()
