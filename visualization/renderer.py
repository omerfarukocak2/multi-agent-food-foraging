"""Pygame dashboard. Rendering and animation never modify simulation rules."""
from collections import deque
from pathlib import Path
import math
import numpy as np
import pygame
from config import Config
from environment import Environment
from evaluation.evaluate import make_policies

BG = (13, 19, 32)
PANEL = (24, 33, 49)
MUTED = (143, 160, 182)
TEXT = (231, 239, 249)
GREEN = (64, 214, 156)
RED = (245, 105, 125)
AGENT_COLORS = [(79, 177, 255), (255, 193, 89), (183, 143, 255)]


class SimulationRenderer:
    def __init__(self, seed=42, mode='mixed', model_dir=Path('models/run_0'), steps=500):
        pygame.init()
        self.screen = pygame.display.set_mode((1260, 890))
        pygame.display.set_caption('BIL353 | Multi-Agent Food Foraging')
        self.font = pygame.font.SysFont('segoeui', 18)
        self.small = pygame.font.SysFont('segoeui', 14)
        self.heading = pygame.font.SysFont('segoeui', 31, bold=True)
        self.medium = pygame.font.SysFont('segoeui', 22, bold=True)
        self.clock = pygame.time.Clock()
        self.model_dir, self.mode, self.seed = Path(model_dir), mode, seed
        self.config = Config(max_steps=steps)
        self.env = Environment(self.config)
        self.running = False
        self.trails = True
        self.speed = 8
        self.elapsed = 0.
        self.message = 'Ready. Press Space or click Start.'
        self.buttons = {}
        self.reset()

    def reset(self, new_seed=False):
        if new_seed:
            self.seed += 1
        self.obs = self.env.reset(self.seed)
        names = ['random', 'heuristic', 'rl'] if self.mode == 'mixed' else [self.mode] * 3
        try:
            self.policies = make_policies(names, self.seed, self.model_dir)
        except (FileNotFoundError, ValueError, KeyError) as error:
            self.mode = 'heuristic'
            self.policies = make_policies(['heuristic'] * 3, self.seed, self.model_dir)
            self.message = f'Model unavailable. Heuristic loaded. {type(error).__name__}'
        self.paths = [[a.position] for a in self.env.agents]
        self.traces = [deque([a.position], maxlen=100) for a in self.env.agents]
        self.actions = [4, 4, 4]
        self.elapsed = 0.

    def advance(self):
        if self.env.done:
            self.running = False
            self.message = 'Episode complete. Reset or try a new map.'
            return
        self.actions = [p.act(o) for p, o in zip(self.policies, self.obs)]
        self.obs, _, _, info = self.env.step(self.actions)
        self.paths = info['paths']
        for trace, path in zip(self.traces, self.paths):
            trace.extend(path[1:])

    def text(self, value, xy, color=TEXT, font=None):
        self.screen.blit((font or self.font).render(str(value), True, color), xy)

    def button(self, key, label, rect, active=False):
        rect = pygame.Rect(rect)
        self.buttons[key] = rect
        pygame.draw.rect(self.screen, (43, 87, 91) if active else (38, 51, 72), rect, border_radius=9)
        image = self.small.render(label, True, GREEN if active else TEXT)
        self.screen.blit(image, image.get_rect(center=rect.center))

    @staticmethod
    def pixel(position):
        return (int(30 + float(position[1]) * 14 + 7), int(120 + float(position[0]) * 14 + 7))

    def draw(self, alpha=1.):
        self.screen.fill(BG)
        self.buttons.clear()
        self.text('FORAGE / 03', (30, 22), font=self.heading)
        self.text('BIL353 ARTIFICIAL INTELLIGENCE  /  MULTI-AGENT LAB', (32, 67), MUTED, self.small)
        self.text('50 x 50', (620, 35), GREEN, self.medium)
        pygame.draw.rect(self.screen, PANEL, (22, 112, 716, 716), border_radius=12)
        pygame.draw.rect(self.screen, (17, 26, 40), (30, 120, 700, 700))
        for offset in range(0, 701, 14):
            pygame.draw.line(self.screen, (27, 39, 55), (30 + offset, 120), (30 + offset, 820))
            pygame.draw.line(self.screen, (27, 39, 55), (30, 120 + offset), (730, 120 + offset))
        for row, col in np.argwhere(self.env.board != 0):
            color = GREEN if self.env.board[row, col] == 1 else RED
            pygame.draw.circle(self.screen, color, self.pixel((row, col)), 3)
        if self.trails:
            for trace, color in zip(self.traces, AGENT_COLORS):
                points = [self.pixel(pos) for pos in trace]
                if len(points) > 1:
                    pygame.draw.lines(self.screen, tuple(c // 3 for c in color), False, points, 2)
        for i, (agent, policy, path) in enumerate(zip(self.env.agents, self.policies, self.paths)):
            target = getattr(policy, 'target', None)
            if target is not None:
                center = self.pixel(target)
                pygame.draw.circle(self.screen, AGENT_COLORS[i], center, 8, 1)
                pygame.draw.line(self.screen, tuple(c // 2 for c in AGENT_COLORS[i]),
                                 self.pixel(agent.position), center, 1)
            t = max(0, min(1, alpha)) * (len(path) - 1)
            segment = min(int(t), max(0, len(path) - 2))
            if len(path) > 1:
                p = np.array(path[segment]) * (1 - (t - segment)) + np.array(path[segment + 1]) * (t - segment)
            else:
                p = path[0]
            center = self.pixel(p)
            pygame.draw.circle(self.screen, BG, center, 9)
            pygame.draw.circle(self.screen, AGENT_COLORS[i], center, 7)
            label = self.small.render(str(i + 1), True, BG)
            self.screen.blit(label, label.get_rect(center=center))
        x = 768
        self.text('LIVE EXPERIMENT', (x, 25), font=self.medium)
        self.text(f'Seed {self.seed}   |   {self.env.metric.title()} distance', (x, 61), MUTED, self.small)
        pygame.draw.rect(self.screen, PANEL, (x, 96, 460, 74), border_radius=12)
        self.text(f'{self.env.steps:03d} / {self.config.max_steps}', (x + 15, 106), font=self.medium)
        self.text('DECISION STEPS', (x + 15, 140), MUTED, self.small)
        self.text(f'{sum(a.score for a in self.env.agents):+d}', (x + 285, 106), GREEN, self.medium)
        self.text('TOTAL REWARD', (x + 285, 140), MUTED, self.small)
        for i, (agent, policy) in enumerate(zip(self.env.agents, self.policies)):
            y = 185 + i * 116
            pygame.draw.rect(self.screen, PANEL, (x, y, 460, 104), border_radius=12)
            pygame.draw.rect(self.screen, AGENT_COLORS[i], (x, y + 13, 4, 78), border_radius=2)
            self.text(f'AGENT {i + 1}  /  {policy.name}', (x + 16, y + 9), AGENT_COLORS[i])
            self.text(f'{agent.score:+d}', (x + 371, y + 8), font=self.medium)
            self.text(f'Green {agent.green}  (+{agent.green})     Red {agent.red}  (-{agent.red})',
                      (x + 16, y + 40), font=self.small)
            target = getattr(policy, 'target', None)
            intent = f'Target {target}' if target else 'Action-value policy' if policy.name == 'Double DQN' else 'No fixed target'
            self.text(f'{intent}   |   Speed {agent.speed}', (x + 16, y + 70), MUTED, self.small)
        y = 551
        self.button('play', 'PAUSE' if self.running else 'START', (x, y, 143, 40), self.running)
        self.button('reset', 'RESET', (x + 155, y, 143, 40))
        self.button('new', 'NEW MAP', (x + 310, y, 150, 40))
        for index, mode in enumerate(['mixed', 'random', 'heuristic', 'rl']):
            self.button('mode_' + mode, mode.upper(), (x + index * 117, 605, 109, 36), self.mode == mode)
        self.button('slower', '-', (x, 655, 42, 36))
        self.text(f'{self.speed} steps / second', (x + 56, 660), MUTED)
        self.button('faster', '+', (x + 240, 655, 42, 36))
        self.button('trails', 'TRAILS', (x + 302, 655, 158, 36), self.trails)
        self.button('load', 'LOAD NEXT TRAINED MODEL', (x, 705, 460, 38))
        self.text(f'Model: {self.model_dir.name}', (x, 756), MUTED, self.small)
        self.text(f'Food left: {np.count_nonzero(self.env.board == 1)} green / '
                  f'{np.count_nonzero(self.env.board == -1)} red', (x, 781), MUTED, self.small)
        self.text(f'Food contests {self.env.totals["competitions"]}  |  '
                  f'Left wins {self.env.totals["left_priority_wins"]}', (x, 805), MUTED, self.small)
        self.text('SPACE play/pause    R reset    N new map    T trails    +/- speed    L load',
                  (30, 842), MUTED, self.small)
        self.text(self.message[:66], (768, 842), GREEN, self.small)
        pygame.display.flip()

    def control(self, key):
        if key == 'play':
            self.running = not self.running
        elif key in ('reset', 'new'):
            self.reset(key == 'new')
        elif key == 'trails':
            self.trails = not self.trails
        elif key == 'slower':
            self.speed = max(1, self.speed // 2)
        elif key == 'faster':
            self.speed = min(64, self.speed * 2)
        elif key.startswith('mode_'):
            self.mode = key[5:]
            self.reset()
        elif key == 'load':
            directories = sorted(p for p in self.model_dir.parent.glob('run_*')
                                 if (p / 'agent_0.npz').exists())
            if directories:
                current = directories.index(self.model_dir) if self.model_dir in directories else -1
                self.model_dir = directories[(current + 1) % len(directories)]
                self.reset()
                self.message = f'Loaded {self.model_dir.name}; episode reset.'
            else:
                self.message = 'No saved models found. Run training first.'

    def run(self, frames=None, snapshot=None, demo_steps=0):
        for _ in range(demo_steps):
            self.advance()
        count = 0
        alive = True
        while alive:
            dt = min(self.clock.tick(60) / 1000, .1)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    alive = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for key, rect in list(self.buttons.items()):
                        if rect.collidepoint(event.pos):
                            self.control(key)
                elif event.type == pygame.KEYDOWN:
                    mapping = {pygame.K_SPACE: 'play', pygame.K_r: 'reset', pygame.K_n: 'new',
                               pygame.K_t: 'trails', pygame.K_l: 'load', pygame.K_MINUS: 'slower',
                               pygame.K_EQUALS: 'faster', pygame.K_PLUS: 'faster'}
                    if event.key == pygame.K_ESCAPE:
                        alive = False
                    elif event.key in mapping:
                        self.control(mapping[event.key])
            if self.running:
                self.elapsed += dt
                period = 1 / self.speed
                while self.elapsed >= period and self.running:
                    self.elapsed -= period
                    self.advance()
            alpha = min(1., self.elapsed * self.speed) if self.running else 1.
            self.draw(alpha)
            count += 1
            if frames is not None and count >= frames:
                alive = False
        if snapshot:
            Path(snapshot).parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(self.screen, str(snapshot))
        pygame.quit()
