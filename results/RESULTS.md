# Recorded experimental results

5 independent repeats, 60 unseen maps, 300 steps, no shaping.

| Policy | Mean team reward | 95% CI |
|---|---:|---|
| Random | 1.34 | [-0.52, 3.34] |
| Heuristic | 259.74 | [256.84, 262.67] |
| Double DQN | 234.04 | [216.31, 252.96] |

These are measured results; see summary.json for paired differences.
Training and final test seeds are disjoint. The 500-step GUI default is a different horizon.