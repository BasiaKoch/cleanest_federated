"""Analyse the asymmetric-straggler experiment (Li 2020 §5.2 protocol).

Compares FedAvg-with-stragglers-dropped vs FedProx-with-stragglers-kept
on the engineered ``balanced_paired_7_clients`` partition under random
stragglers (C2). This is the canonical Li 2020 §5.2 evaluation that
produces the literature's headline FedProx advantage.

Reports per-seed FedAvg vs FedProx macro_f1, the within-pair Δ,
formal paired Wilcoxon p-value, and contrasts the result against:
  - The symmetric C2 result (system_het_random/, both algos see same updates)
  - The pure-PyTorch headline (PT runtime baseline)
  - The Flower C0 baseline (uniform compute, no stragglers)

Outputs are not modified; this is a read-only audit script.
"""
from __future__ import annotations

import json
import os
import re
from statistics import mean, median, stdev


ASYM_DIR = "mnist_dermnist/results/system_het_random_asymmetric"
SYM_DIR  = "mnist_dermnist/results/system_het_random"
C0_DIR   = "mnist_dermnist/results/flower_C0_baseline"
PT_DIR   = "mnist_dermnist/results/headline"

# Asymmetric: FedAvg files have _drop tag; FedProx files do not.
_ASYM_PAT = re.compile(
    r"test_at_best_(?P<algo>fedavg|fedprox)_mu[0-9.]+_E20"
    r"_sh-random_stragglers(?:_drop)?_s(?P<seed>\d+)\.json"
)
_STANDARD_PAT = re.compile(
    r"test_at_best_(?P<algo>fedavg|fedprox)_mu[0-9.]+_E20"
    r"(?:_sh-[a-z_]+)?_s(?P<seed>\d+)\.json"
)


def load_dir(d, pattern):
    """Return {algo: {seed: macro_f1}} from a results dir."""
    out = {"fedavg": {}, "fedprox": {}}
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        m = pattern.match(f)
        if not m:
            continue
        algo = m.group("algo")
        seed = int(m.group("seed"))
        doc = json.load(open(os.path.join(d, f)))
        out[algo][seed] = float(doc["macro_f1"])
    return out


def paired_deltas(results):
    seeds = sorted(set(results["fedavg"]) & set(results["fedprox"]))
    return seeds, [results["fedprox"][s] - results["fedavg"][s] for s in seeds]


def print_summary(label, results):
    print(f"\n=== {label} ===")
    fa = results["fedavg"]
    fp = results["fedprox"]
    print(f"  FedAvg : n={len(fa):2d}  mean={mean(fa.values()):.4f}"
          if fa else "  FedAvg: (no data)")
    print(f"  FedProx: n={len(fp):2d}  mean={mean(fp.values()):.4f}"
          if fp else "  FedProx: (no data)")
    seeds, deltas = paired_deltas(results)
    if not deltas:
        return None
    n = len(deltas)
    wins = sum(1 for d in deltas if d > 0)
    mu = mean(deltas)
    med = median(deltas)
    sd = stdev(deltas) if n > 1 else 0.0
    print(f"  Δ (FedProx - FedAvg):  n={n}  mean={mu:+.4f}  median={med:+.4f}  SD={sd:.4f}  wins={wins}/{n}")
    try:
        from scipy.stats import wilcoxon
        if any(d != 0 for d in deltas):
            p = wilcoxon(deltas, alternative="two-sided").pvalue
            print(f"  Paired Wilcoxon two-sided p = {p:.4f}")
    except ImportError:
        pass
    return mu


def main():
    print("=" * 80)
    print(" ASYMMETRIC STRAGGLER PROTOCOL (Li 2020 §5.2)")
    print("=" * 80)

    asym = load_dir(ASYM_DIR, _ASYM_PAT)
    asym_mean = print_summary("ASYMMETRIC: FedAvg-drops-stragglers vs "
                              "FedProx-includes-stragglers", asym)

    sym = load_dir(SYM_DIR, _STANDARD_PAT)
    sym_mean = print_summary("SYMMETRIC (current): both algos see all updates",
                             sym)

    c0 = load_dir(C0_DIR, _STANDARD_PAT)
    c0_mean = print_summary("C0 (uniform compute, no stragglers)", c0)

    pt = load_dir(PT_DIR, _STANDARD_PAT)
    pt_mean = print_summary("Pure-PyTorch headline (no stragglers)", pt)

    print("\n" + "=" * 80)
    print(" CROSS-PROTOCOL CONTRAST")
    print("=" * 80)
    rows = [
        ("Pure-PyTorch headline (no stragglers, no Flower)", pt_mean),
        ("Flower C0 (uniform compute)",                     c0_mean),
        ("Flower C2 symmetric (current)",                   sym_mean),
        ("Flower C2 ASYMMETRIC (Li 2020 §5.2)",             asym_mean),
    ]
    print(f"\n  {'Condition':<55s}  {'mean Δ':>10s}")
    print("  " + "-" * 70)
    for label, m in rows:
        m_str = f"{m:+.4f}" if m is not None else "(no data)"
        print(f"  {label:<55s}  {m_str:>10s}")
    print()
    print("  Interpretation:")
    print("  - If ASYMMETRIC >> SYMMETRIC: the FedProx advantage reported in the")
    print("    literature is largely attributable to differential straggler handling,")
    print("    NOT the proximal-anchor mechanism per se.")
    print("  - If ASYMMETRIC ≈ SYMMETRIC: the straggler-dropping protocol is not the")
    print("    source of the literature's larger effect sizes on this dataset.")


if __name__ == "__main__":
    raise SystemExit(main())
