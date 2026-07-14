"""run_qpu.py -- the same experiments on real D-Wave hardware.

Prerequisites (on your machine, not in a sandbox):
    pip install dwave-ocean-sdk
    dwave auth login                # or: export DWAVE_API_TOKEN=DEV-xxxx
    dwave ping                      # verify solver access

Backends:
    --backend qpu     : EmbeddingComposite(DWaveSampler())  -- direct QPU.
                        Suitable for every TSP BQM in this repo (<= ~200
                        variables) and for assignment BQMs up to roughly
                        150-200 variables (dense); beyond that embedding
                        fails or chains get long.
    --backend hybrid  : LeapHybridBQMSampler -- no size limit; uses Leap
                        hybrid solver time. Suitable for the full-size
                        assignment BQMs of exp_full (up to ~1000 vars).
    --backend sa      : dwave.samplers SA (what the bundled experiments
                        used) -- for parity checks.

Examples:
    python run_qpu.py --exp small --backend qpu   --reads 1000
    python run_qpu.py --exp full  --backend hybrid
    python run_qpu.py --exp ablation --backend qpu --reads 2000

The QPU-relevant metrics to compare against results_exp*.csv:
  - valid-sample fraction per encoding (domain-wall is EXPECTED to close
    or reverse its SA gap on hardware: its 1-D chain constraint structure
    embeds with shorter chains than one-hot's cliques -- this is the
    hypothesis the ablation is designed to test),
  - chain break fraction (ss.record.chain_break_fraction),
  - best feasible distance at equal QPU time.
"""
from __future__ import annotations

import argparse
import sys


def get_sampler(backend: str):
    if backend == "sa":
        from dwave.samplers import SimulatedAnnealingSampler
        return SimulatedAnnealingSampler(), "classical-sa"
    if backend == "qpu":
        from dwave.system import DWaveSampler, EmbeddingComposite
        s = DWaveSampler()
        print(f"QPU solver: {s.solver.name}")
        return EmbeddingComposite(s), "qpu"
    if backend == "hybrid":
        from dwave.system import LeapHybridBQMSampler
        return LeapHybridBQMSampler(), "hybrid"
    raise ValueError(backend)


def patch_samplers(sampler, kind, reads):
    """Monkey-patch samplers.sample_bqm so every experiment script runs
    unchanged on the chosen backend."""
    import time

    import samplers as S

    def sample_bqm(bqm, backend="ignored", num_reads=None, sweeps=None,
                   seed=None):
        t0 = time.time()
        n = num_reads or reads
        if kind == "qpu":
            ss = sampler.sample(bqm, num_reads=n, label="p1quantum")
            cbf = float(ss.record.chain_break_fraction.mean()) \
                if "chain_break_fraction" in ss.record.dtype.names else -1
            print(f"    [qpu] {bqm.num_variables} vars, "
                  f"chain-break frac = {cbf:.3f}")
        elif kind == "hybrid":
            ss = sampler.sample(bqm, label="p1quantum")
        else:
            ss = sampler.sample(bqm, num_reads=n,
                                num_sweeps=sweeps or 2000, seed=seed)
        return ss, time.time() - t0

    S.sample_bqm = sample_bqm
    # experiment modules import the symbol directly; patch there too
    for mod in ("exp_small", "exp_full", "exp_ablation"):
        if mod in sys.modules:
            sys.modules[mod].sample_bqm = sample_bqm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["small", "full", "ablation"],
                    required=True)
    ap.add_argument("--backend", choices=["qpu", "hybrid", "sa"],
                    default="qpu")
    ap.add_argument("--reads", type=int, default=1000)
    ap.add_argument("files", nargs="*")
    args = ap.parse_args()

    sampler, kind = get_sampler(args.backend)

    if args.exp == "small":
        import exp_small
        patch_samplers(sampler, kind, args.reads)
        exp_small.sample_bqm = sys.modules["samplers"].sample_bqm
        sys.argv = ["exp_small.py"] + args.files
        exp_small.main()
    elif args.exp == "full":
        import exp_full
        patch_samplers(sampler, kind, args.reads)
        exp_full.sample_bqm = sys.modules["samplers"].sample_bqm
        sys.argv = ["exp_full.py"] + args.files
        exp_full.main()
    else:
        import exp_ablation
        patch_samplers(sampler, kind, args.reads)
        exp_ablation.sample_bqm = sys.modules["samplers"].sample_bqm
        exp_ablation.main()


if __name__ == "__main__":
    main()
