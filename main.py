"""Interactive simulation, or a reproducible headless dashboard snapshot."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', choices=['mixed', 'random', 'heuristic', 'rl'], default='mixed')
    parser.add_argument('--model-dir', type=Path, default=Path('models/run_0'))
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--frames', type=int)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--demo-steps', type=int, default=0)
    args = parser.parse_args()
    if args.headless:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
        if args.frames is None:
            args.frames = 1
    from visualization.renderer import SimulationRenderer
    SimulationRenderer(args.seed, args.mode, args.model_dir, args.steps).run(
        args.frames, args.snapshot, args.demo_steps)


if __name__ == '__main__':
    main()
