# Environment contract (version 1)

These are implementation assumptions where the supplied assignment is ambiguous,
not claims that the instructor explicitly specified these details. The assignment
requirements copied into the project are derived from the user's supplied brief.

1. Board: 50 rows x 50 columns. Coordinates are zero-based `(row, column)`.
2. Exactly three agents, initially at distinct food-free positions. Food cells are
   sampled without replacement. Default: 375 green and 375 red (30% occupancy).
3. All remaining food is visible. No other agent occupancy is encoded in the food
   board. Each policy receives its own position and current speed, exactly one
   nearest neighbor's position, the metric and normalized remaining horizon.
   Neighbor speed, reward, identity, actions and targets are not observations.
4. Manhattan or Euclidean nearest-neighbor metric is sampled with equal probability
   at reset and remains fixed during that episode. Equal nearest distances use a
   deterministic seeded random choice; observation calls do not advance RNG state.
5. Each agent's speed is independently uniform in {1,2,3} before every decision,
   and visible to that agent before it acts. Speed is indexed by map seed and step.
6. All agents act on the same pre-transition state. Straight paths are executed
   synchronously in at most three substeps. Every visited food cell is eligible.
7. Bounds stop movement at the last valid cell, without a penalty. Positions can
   overlap and agents can pass or swap. These interactions are counted, not blocked.
8. A food is consumed once. For simultaneous claims to a green in the SAME substep,
   Left action has priority. Earlier arrival in a previous substep wins immediately.
   Multiple Left contenders, no Left contender, or simultaneous red claims use a
   seeded random winner. A loser is not moved elsewhere and receives zero.
9. Each consumed green gives its consumer +1; each red gives its consumer -1. An
   action can consume up to three foods. No movement, waiting or collision reward,
   no shaping and no shared team training reward are added.
10. No respawn. End when all food is consumed or the configured decision horizon is
    reached. Default interactive horizon: 500; delivered experiment horizon: 300.
    The horizon is part of the finite task; DQN stops bootstrapping at both endings.
11. No Action stays still. It does not repeatedly claim already consumed food.
12. Renderer sees the whole world as a spectator; policies cannot access renderer,
    environment state or the full agent list.

## Metrics

- Team reward: sum of all three individual episode rewards, reported for evaluation
  only. It is never the training reward for an individual agent.
- Individual green rate: individual green count / initial green count.
- Team green rate: sum of individual green counts / initial green count (0..1).
- Overlaps: one event per agent pair per substep sharing an endpoint, including
  stationary overlaps. This is occupancy frequency, not a physical collision count.
- Swaps: one event per pair per substep exchanging distinct starting positions.
- Competition: one event per food cell with two or more simultaneous claimants.
- Left priority wins: green competitions with at least one eligible Left contender.
- Boundary hits: attempted outward movement, counted once per agent per decision.

## Reproducibility

Independent SeedSequence namespaces: layout=11, metric=21, speed=31,
nearest-neighbor ties=61 and food arbitration=71. Policy randomness has its own
generator. Food arbitration is keyed by (seed, step, substep, cell), so an extra
competition cannot change future speeds. Repeatability requires the same config,
seed, actions and compatible dependency versions. Cross-platform neural floating
point arithmetic may differ; a seed alone is not an across-hardware bitwise guarantee.

## Instructor clarification priorities

Confirm speed resampling frequency, food collection along intermediate cells,
same-cell occupancy and the temporal interpretation of Left priority before final
submission. If clarified differently, change the contract, implementation and
tests together, and rerun experiments. No requirement point allocation is invented.
