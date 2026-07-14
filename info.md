# p1quantum — optimizing paper 1's QUBO formulation, with sampling experiments

Target paper: **Borowski et al., *New Hybrid Quantum Annealing Algorithms
for Solving Vehicle Routing Problem*, ICCS 2020** (FQS / APS / DBSS / SPS).

Improved stack (all zero-slack-qubit techniques, composed):
- **Decomposition**: monolithic `x_{v,j,k}` (O(M N^2) vars) -> assignment
  BQM (N*M) + independent per-cluster TSP BQMs (n(n-1) or n^2).
- **Unbalanced penalization** for capacity (no slack binaries).
- **TW conflict-pair penalties** in the TSP stage AND (new in this round)
  mutual-conflict penalties in the assignment stage — time-window structure
  enters both QUBOs at zero qubit cost.
- **Domain-wall encoding** as an exact affine transform of one-hot
  (verified to 5.6e-12 over all tours; `tests_encoding.py`).
- **Exact-window coarsening** (round-2 result) as the front end for
  full-size instances.
- **Sample-pool utilization**: route the top-K distinct assignment samples,
  keep the best feasible end-to-end solution.

## Execution environment — read this first

All bundled experiments were run with **`dwave.samplers.SimulatedAnnealingSampler`**
(D-Wave's production classical annealer, dimod-native) — real sampling
experiments, but **classical**. Actual QPU access requires a D-Wave Leap
token, which this environment does not have. `run_qpu.py` runs the *same
three experiments unchanged* on real hardware (`--backend qpu` for direct
embedding, `--backend hybrid` for Leap hybrid) once you `dwave auth login`.
Leap's free tier includes enough QPU time for exp_small and exp_ablation.

## Experiment A — head-to-head at annealer scale (results_expA.csv)

Truncated Solomon instances at N = 5, 10 (the sizes of paper 3's quantum
experiments); fleet sized by TW-aware greedy; identical sampling budget
(500 reads x 2000 sweeps); EXACT-OPT = brute force (N=5 only).

| instance | N | method | vars | max BQM | best feasible dist |
|---|---|---|---|---|---|
| C101 | 5 | EXACT-OPT | — | — | 42.4 |
| C101 | 5 | FQS | 50 | 50 | 72.7 |
| C101 | 5 | DECOMP-DW | 50 | 10 | 74.7 |
| C101 | 10 | FQS | 300 | 300 | **infeasible** |
| C101 | 10 | DECOMP-DW | 162 | 30 | 123.9 |
| R101 | 5 | EXACT-OPT | — | — | 156.3 |
| R101 | 5 | FQS | 75 | 75 | 156.4 |
| R101 | 5 | DECOMP-DW | 24 | 15 | 156.6 |
| R101 | 10 | FQS | 600 | 600 | **infeasible** |
| R101 | 10 | DECOMP-DW | 99 | 60 | 312.1 |
| RC101 | 10 | FQS | 500 | 500 | 410.3 |
| RC101 | 10 | DECOMP-DW | 120 | 50 | 374.7 |

Reading: at N=5 FQS is competitive (decomposition overhead not worth it at
toy size). At N=10 FQS returns structurally valid but TW-infeasible
solutions on 2/3 instances (it has no TW terms) and is beaten where
feasible, while needing 5-10x more variables per BQM. The mutual-conflict
assignment penalty was decisive on R101 (inf -> 156.6, a 0.2% gap to the
exact optimum). Note: R101-N10 with 2 vehicles was PROVEN infeasible during
debugging — fleet sizes must come from TW-aware bounds, not capacity bounds.

## Experiment B — full Solomon-100 hybrid pipeline (results_expB.csv)

Exact-window coarsening (P=0.4) -> conflict-aware assignment BQM (sampled)
-> per-cluster domain-wall TSP BQMs (sampled) -> inflation, NO repair.
Fleet sized and pool warm-started by a coarse-level savings run.

| instance | max BQM | FQS-equivalent vars | dist / veh | feasible | vs full-instance savings | best from |
|---|---|---|---|---|---|---|
| C101 | 560 | 140,000 | 1192.3 / 13 | yes | 930.1 / 12 (worse) | warm start |
| R101 | 1000 | 250,000 | 1959.8 / 24 | yes | 2002.4 / 31 (**better**) | warm start |
| RC101 | 920 | 230,000 | 2125.3 / 22 | yes | 2134.9 / 26 (**better**) | warm start |
| C201 | 280 | 70,000 | 1686.5 / 7 | yes | 751.8 / 6 (worse) | **qubo-sampled** |

Honest reading: every BQM is 250-4500x smaller than the FQS equivalent and
all solutions are fully feasible with zero repair (the round-2 guarantee at
work). On R/RC geometries the pipeline beats full-instance savings on both
distance and vehicles. BUT at this fleet size clusters average <2
super-nodes, so in 3/4 cases the warm-start clustering was not beaten by
QUBO sampling — the QUBO stages earn their keep at Experiment-A scale, not
here. Claiming otherwise would not survive review.

## Experiment C — encoding ablation under SA (results_expC.csv)

Open TSPs (n = 6..14, 3 trials each, equal budget):

| n | one-hot valid% | domain-wall valid% | vars saved | quality |
|---|---|---|---|---|
| 6 | 100.0 | ~99.7 | 36 -> 30 | tie |
| 8 | 100.0 | ~96.5 | 64 -> 56 | tie |
| 10 | 100.0 | ~89.0 | 100 -> 90 | tie |
| 12 | 100.0 | ~76.0 | 144 -> 132 | tie |
| 14 | 100.0 | ~60.1 | 196 -> 182 | tie |

**Under classical SA, domain-wall's valid-sample rate degrades with size**
while best-tour quality stays comparable. The documented domain-wall
advantage (Chen-Stollenwerk-Chancellor 2021) is a *hardware-embedding*
effect (1-D chain constraints embed with shorter chains than one-hot
cliques); SA has no embedding. This is exactly the hypothesis
`run_qpu.py --exp ablation --backend qpu` tests — compare valid% and
chain-break fraction on real hardware before choosing the encoding for QPU
runs.

## Files

- `qubo.py`            — FQS baseline + improved builders (dimod-native)
- `samplers.py`        — D-Wave sampler layer (SA/tabu/steepest)
- `exp_small.py`       — Experiment A (+ brute-force exact optimum)
- `exp_full.py`        — Experiment B
- `exp_ablation.py`    — Experiment C
- `tests_encoding.py`  — encoding exactness proofs-by-exhaustion
- `run_qpu.py`         — same experiments on Leap QPU / hybrid
- `coarsen.py`, `solvers.py`, `instances.py` — round-2 exact-window
  coarsener and classical references
- `results_expA/B/C.csv` — the raw results reported above
- Solomon CSVs (from the paper-3 authors' repository)

## How to run

```bash
pip install dimod dwave-samplers        # numpy assumed
python tests_encoding.py                # must print ALL ... PASSED
python exp_small.py                     # ~1 min
python exp_full.py                      # ~2 min
python exp_ablation.py                  # ~1 min

# real hardware (your machine, Leap account):
pip install dwave-ocean-sdk && dwave auth login
python run_qpu.py --exp small    --backend qpu    --reads 1000
python run_qpu.py --exp ablation --backend qpu    --reads 2000
python run_qpu.py --exp full     --backend hybrid
```

## What to do next (priority order)

1. QPU runs of exp_ablation (cheapest, most decisive: settles the encoding
   question with chain-break data).
2. QPU runs of exp_small (direct FQS-vs-decomposed on hardware at N=10,
   where FQS barely embeds and the decomposed BQMs embed trivially).
3. Penalty-weight sensitivity sweep (all weights here are heuristic).
4. Larger sampling budgets + a Benders-style loop feeding realized route
   costs back into the assignment BQM (the Exp-B gap to warm-start
   clustering is the target).
