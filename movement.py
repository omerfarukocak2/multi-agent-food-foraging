"""Synchronous substep movement and atomic food arbitration."""
import numpy as np
from agent import Action, DELTAS
from rewards import RewardManager


class MovementResolver:
    @staticmethod
    def resolve(environment, actions: list[int]):
        rewards = np.zeros(3, dtype=np.float32)
        events = {"overlaps": 0, "swaps": 0, "competitions": 0,
                  "green_competitions": 0, "left_priority_wins": 0,
                  "boundary_hits": 0}
        paths = [[a.position] for a in environment.agents]
        stopped = [False] * 3
        for substep in range(max(a.speed for a in environment.agents)):
            old = [a.position for a in environment.agents]
            proposed = old.copy()
            for i, agent in enumerate(environment.agents):
                if substep >= agent.speed or stopped[i] or actions[i] == Action.NO_ACTION:
                    continue
                dr, dc = DELTAS[actions[i]]
                candidate = (agent.position[0] + dr, agent.position[1] + dc)
                if all(0 <= coordinate < 50 for coordinate in candidate):
                    proposed[i] = candidate
                else:
                    events["boundary_hits"] += 1
                    stopped[i] = True
            for i in range(3):
                for j in range(i + 1, 3):
                    events["overlaps"] += int(proposed[i] == proposed[j])
                    events["swaps"] += int(old[i] != old[j] and proposed[i] == old[j]
                                           and proposed[j] == old[i])
            claims: dict[tuple[int, int], list[int]] = {}
            for i, agent in enumerate(environment.agents):
                agent.position = proposed[i]
                paths[i].append(agent.position)
                if environment.board[agent.position] != 0:
                    claims.setdefault(agent.position, []).append(i)
            for cell, contenders in sorted(claims.items()):
                food = int(environment.board[cell])
                eligible = contenders
                if len(contenders) > 1:
                    events["competitions"] += 1
                    if food == 1:
                        events["green_competitions"] += 1
                        left = [i for i in contenders if actions[i] == Action.LEFT]
                        if left:
                            eligible = left
                            events["left_priority_wins"] += 1
                # Event-keyed randomization cannot shift later speeds or maps.
                rng = np.random.default_rng(np.random.SeedSequence(
                    [environment.seed, 71, environment.steps, substep, *cell]))
                winner = int(rng.choice(eligible)) if len(eligible) > 1 else eligible[0]
                rewards[winner] += RewardManager.collect(environment.agents[winner], food)
                environment.board[cell] = 0
        return rewards, events, paths
