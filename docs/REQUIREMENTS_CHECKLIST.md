# Requirement-to-evidence map

This is a checklist against the supplied brief, not an official marking rubric.
The requested 80+20 split is treated as core requirements and advanced evidence;
no individual point allocations or guaranteed grade are invented.

| Requirement | Implementation | Evidence |
|---|---|---|
| 50x50 board; exactly 3 agents | config.py, environment.py | test_initial_invariants |
| Equal green/red, at most 30% initial occupancy | Environment.reset | invariant and invalid-config tests |
| Random valid initial positions | food-free permutation cells | invariant tests over 10 seeds |
| Observable food | 2500 board entries in observation | observation equality assertions |
| Only nearest other position | ObservationManager | hidden-agent invariance and metric tests |
| Random Manhattan/Euclidean method | SeedSequence namespace 21, equal selection | metric support, distance tests, CSV metric column |
| Random speed 1..3 | namespace 31, per decision | speed support and paired-scenario tests |
| Five actions and vector observation | Action, 2508 float32 vector | five parameterized action cases |
| Green +1, red -1, independent rewards | RewardManager | path, red competition and conservation tests |
| Left wins green competition | MovementResolver | priority tests for all three agent IDs |
| Bounds and no duplicate collection | synchronous claims | boundaries, earlier arrival, conservation |
| Modular object-oriented Python, seeds and errors | config / environment / policy / manager classes | tests, config records and locked dependencies |
| Modern Pygame board, agent colors, trails, animation | SimulationRenderer | dashboard.png and control smoke test |
| Targets, scores, counts, steps, algorithms, speed | dashboard panels; true heuristic target | dashboard.png; RL explicitly uses action values |
| Start/pause/reset/model load | buttons and keyboard controls | renderer tests, model tests and 15-model load verification |
| Headless training | training.train | 5 training CSVs and selected/final NPZ files |
| Random baseline | RandomAgent | homogeneous and mixed test rows |
| Intelligent heuristic | risk-aware reverse Dijkstra planner | red-wall routing test and held-out results |
| Reinforcement learning | Independent Double DQN with 3 separate networks | learning curves, supervised/update tests |
| Exploration/exploitation | epsilon schedule | per-episode training CSV |
| Learned model save/load | versioned NPZ, no pickle | roundtrip and all-model load verification |
| New maps, shared competitive environment | separate train/validation/test seeds | evaluation logs and protocol in report |
| All requested performance metrics | ExperimentEvaluator | episodes.csv and summary.json |
| Five requested graph categories | evaluation.plots | six PNG/PDF plot pairs |
| Repeats, variability and confidence intervals | 5 runs, 60 maps, crossed bootstrap | summary paired differences and report |
| English academic report, 13 sections, equations, references | report.build_reviewed_report | output/pdf/Project_Report_Revised.pdf and report/Project_Report_Revised.md |
| Installation, tests and reproducibility commands | README, Setup.bat, Start_Game.bat | locked packages, runtime and manifest |

## Advanced algorithm evidence

Learning uses only individual, unshaped +1/-1 rewards. Independent learners share
the competitive environment, not parameters, replay or hidden position data.
Exploration, experience replay, target updates and validation checkpoint selection
are recorded. Success claims are limited to measured conditions and intervals.

## Assumptions requiring instructor comparison

Speed timing, intermediate food collection, physical overlap, red arbitration,
Left priority timing and no respawn are explicit implementation assumptions.
See ASSUMPTIONS.md. The original instructor document was not independently supplied.
