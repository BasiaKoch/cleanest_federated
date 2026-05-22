"""μ-sensitivity audit for FedProx on the engineered partition.

Reads μ-sweep results from ``results/mu_sweep/`` plus the existing
headline values at μ=0.01 (FedProx) and μ=0.0 (FedAvg). Reports
Δ_FedProx-FedAvg at each μ value for the 3 ablation seeds (42, 123, 456)
and tells you whether the headline μ=0.01 is at a knife-edge maximum.

Idempotent. Re-run any time:
    python mnist_dermnist/scripts/check_mu_sweep.py
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from statistics import mean, stdev


MU_SWEEP_DIR  = "mnist_dermnist/results/mu_sweep"
HEADLINE_DIR  = "mnist_dermnist/results/headline"
ABLATION_SEEDS = [42, 123, 456]
HEADLINE_MU = 0.01

# Matches both mu_sweep/ and headline/ filenames.
_PAT = re.compile(
    r"test_at_best_(?P<algo>fedavg|fedprox)_mu(?P<mu>[0-9.]+)_E20_s(?P<seed>\d+)\.json"
)


def _safe_float_mu(s):
    """Parse a mu string from a filename, tolerating odd formatting."""
    try:
        return float(s)
    except ValueError:
        return None


def load_results(dir_path, seeds):
    """Return {(algo, mu, seed): macro_f1}."""
    out = {}
    if not os.path.isdir(dir_path):
        return out
    for f in sorted(os.listdir(dir_path)):
        m = _PAT.match(f)
        if not m:
            continue
        algo = m.group("algo")
        mu = _safe_float_mu(m.group("mu"))
        seed = int(m.group("seed"))
        if mu is None or seed not in seeds:
            continue
        doc = json.load(open(os.path.join(dir_path, f)))
        out[(algo, mu, seed)] = float(doc["macro_f1"])
    return out


def main():
    print("=" * 80)
    print(" μ-SENSITIVITY SWEEP — pure-PyTorch FedProx on engineered partition")
    print("=" * 80)

    seeds_set = set(ABLATION_SEEDS)
    sweep_data = load_results(MU_SWEEP_DIR,  seeds_set)
    head_data  = load_results(HEADLINE_DIR,  seeds_set)

    combined = {}
    combined.update(head_data)
    combined.update(sweep_data)

    if not sweep_data:
        print(f"\n  (no μ-sweep data yet — run bash mnist_dermnist/scripts/hpc_mu_sweep.sh)")

    # Aggregate by μ value
    mus = sorted(set(mu for (_, mu, _) in combined.keys() if _ != "ignore"))
    # We compare to FedAvg at μ=0.0 (the algorithmic FedAvg baseline)
    fedavg_seeds = {s: combined.get(("fedavg", 0.0, s))
                    for s in ABLATION_SEEDS}

    print(f"\n FedAvg baseline (μ=0.0) at seeds {ABLATION_SEEDS}:")
    for s in ABLATION_SEEDS:
        val = fedavg_seeds.get(s)
        print(f"   seed {s:>6d}: macro_f1 = {val if val is not None else 'MISSING'}")
    if any(v is None for v in fedavg_seeds.values()):
        print("\n  ERROR: missing FedAvg baseline values; cannot compute Δ.")
        return 2

    # Per-μ deltas
    print(f"\n FedProx μ-sweep — per-seed Δ = FedProx − FedAvg (paired by seed)")
    print()
    print(f"   {'μ':>8s}  {'seed':>6s}  {'FedAvg':>8s}  {'FedProx':>8s}  {'Δ':>9s}")
    print("   " + "-" * 50)

    mu_summary = defaultdict(list)
    fp_mus = sorted(set(mu for (a, mu, _) in combined.keys() if a == "fedprox"))
    for mu in fp_mus:
        for s in ABLATION_SEEDS:
            fa = fedavg_seeds[s]
            fp = combined.get(("fedprox", mu, s))
            if fp is None:
                print(f"   {mu:>8.3f}  {s:>6d}  {fa:>8.4f}  {'?':>8s}  {'?':>9s}")
                continue
            d = fp - fa
            mu_summary[mu].append(d)
            print(f"   {mu:>8.3f}  {s:>6d}  {fa:>8.4f}  {fp:>8.4f}  {d:>+9.4f}")
        if len(mu_summary[mu]) == len(ABLATION_SEEDS):
            mu_mean = mean(mu_summary[mu])
            mu_sd = stdev(mu_summary[mu]) if len(mu_summary[mu]) > 1 else 0.0
            print(f"   {mu:>8.3f}  {'mean':>6s}                         {mu_mean:>+9.4f}  (SD={mu_sd:.4f})")
        print()

    # Aggregate summary
    print()
    print("=" * 80)
    print(" SUMMARY — mean Δ across 3 ablation seeds, per μ")
    print("=" * 80)
    print()
    print(f"   {'μ':>8s}  {'n':>3s}  {'mean Δ':>10s}  {'SD':>8s}  {'wins':>6s}")
    print("   " + "-" * 50)
    sorted_mus = sorted(mu_summary.keys())
    for mu in sorted_mus:
        ds = mu_summary[mu]
        n = len(ds)
        if n == 0:
            continue
        wins = sum(1 for d in ds if d > 0)
        m = mean(ds)
        s = stdev(ds) if n > 1 else 0.0
        marker = "  ←  HEADLINE" if abs(mu - HEADLINE_MU) < 1e-9 else ""
        print(f"   {mu:>8.3f}  {n:>3d}  {m:>+10.4f}  {s:>8.4f}  {wins:>3d}/{n:<2d}{marker}")

    # Interpretation
    print()
    print("=" * 80)
    print(" INTERPRETATION GUIDE")
    print("=" * 80)
    print("   * If mean Δ stays roughly the same across all μ ∈ {0.001, 0.01, 0.1, 1.0}:")
    print("     the headline result is ROBUST to μ choice (most defensible).")
    print()
    print("   * If mean Δ peaks at μ=0.01: the headline μ is well-tuned BUT")
    print("     a reviewer can argue 'why this specifically?'.")
    print()
    print("   * If mean Δ is LARGER at a different μ (especially μ=0.1 or μ=1.0):")
    print("     the headline UNDER-reports FedProx's true advantage on this dataset.")
    print("     This is interesting but requires a re-frame of the headline claim.")
    print()
    print("   * If mean Δ varies wildly: the result is μ-sensitive and the headline")
    print("     value's choice would need stronger pre-registration justification.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
