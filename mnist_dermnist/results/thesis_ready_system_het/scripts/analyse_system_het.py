"""Analyse the system-heterogeneity HPC results.

Reads `test_at_best_*.json` files from
  - mnist_dermnist/results/system_het_fixed/
  - mnist_dermnist/results/system_het_random/
and aggregates them into per-condition paired test statistics (H1) plus
between-condition tests of whether system heterogeneity amplifies the
FedProx advantage (H2).

Compares against the headline statistical-heterogeneity numbers (C0) from
mnist_dermnist/results/headline/ for the H2 test.

Outputs:
  - thesis_ready_system_het/data/per_seed_results.csv
  - thesis_ready_system_het/data/per_class_results.csv
  - thesis_ready_system_het/data/system_het_vs_baseline.json (H2 test)
  - thesis_ready_system_het/data/summary_statistics.json
  - prints a complete results table to stdout
"""
from __future__ import annotations

import json
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


ROOT = Path(__file__).resolve().parent.parent  # thesis_ready_system_het/
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_ROOT = ROOT.parent  # mnist_dermnist/results/


CLASS_NAMES = ["actinic", "basal", "benign_kerat", "dermato",
               "melanoma", "mel_nevi", "vascular"]


def load_pairs(results_dir: Path):
    """Return (fedavg_by_seed, fedprox_by_seed) dicts from a results dir."""
    fa, fp = {}, {}
    # Filename patterns:
    # - headline:           test_at_best_(fedavg|fedprox)_mu(0.0|0.01)_E20_s(\d+).json
    # - system-het fixed:   test_at_best_(fedavg|fedprox)_mu*_E20_sh-fixed_stragglers_s(\d+).json
    # - system-het random:  test_at_best_(fedavg|fedprox)_mu*_E20_sh-random_stragglers_s(\d+).json
    pat = re.compile(
        r"test_at_best_(fedavg|fedprox)_mu[0-9.]+_E20(?:_sh-[a-z_]+)?_s(\d+)\.json"
    )
    for f in sorted(results_dir.glob("test_at_best_*.json")):
        m = pat.match(f.name)
        if not m:
            print(f"  skipped (filename mismatch): {f.name}")
            continue
        algo, seed = m.group(1), int(m.group(2))
        (fa if algo == "fedavg" else fp)[seed] = json.load(open(f))
    return fa, fp


def wilcoxon(deltas):
    if not HAS_SCIPY or all(d == 0 for d in deltas):
        return float("nan")
    try:
        _, p = stats.wilcoxon(deltas, alternative="two-sided")
        return float(p)
    except ValueError:
        return float("nan")


def rank_biserial(deltas):
    if not HAS_SCIPY:
        return float("nan")
    abs_ranks = stats.rankdata([abs(d) for d in deltas])
    pos = sum(abs_ranks[i] for i, d in enumerate(deltas) if d > 0)
    neg = sum(abs_ranks[i] for i, d in enumerate(deltas) if d < 0)
    return (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0.0


def summarise_condition(fa, fp, condition_name, baseline_fa=None, baseline_fp=None):
    seeds = sorted(set(fa) & set(fp))
    n = len(seeds)
    if n == 0:
        print(f"  WARNING: no paired seeds for {condition_name}")
        return None

    deltas = [fp[s]["macro_f1"] - fa[s]["macro_f1"] for s in seeds]
    fa_vals = [fa[s]["macro_f1"] for s in seeds]
    fp_vals = [fp[s]["macro_f1"] for s in seeds]

    out = {
        "condition": condition_name,
        "n_paired_seeds": n,
        "seeds": seeds,
        "fedavg_mean":  float(np.mean(fa_vals)),
        "fedavg_sd":    float(np.std(fa_vals, ddof=1)) if n > 1 else 0.0,
        "fedprox_mean": float(np.mean(fp_vals)),
        "fedprox_sd":   float(np.std(fp_vals, ddof=1)) if n > 1 else 0.0,
        "delta_mean":   float(np.mean(deltas)),
        "delta_sd":     float(np.std(deltas, ddof=1)) if n > 1 else 0.0,
        "fedprox_wins": int(sum(1 for d in deltas if d > 0)),
        "wilcoxon_p_h1":  wilcoxon(deltas),
        "rank_biserial":  rank_biserial(deltas),
        "per_seed_delta": deltas,
    }

    # H2: is this condition's per-seed delta different from baseline's?
    if baseline_fa is not None and baseline_fp is not None:
        common = sorted(set(seeds) & set(baseline_fa) & set(baseline_fp))
        if common:
            baseline_deltas = [baseline_fp[s]["macro_f1"] - baseline_fa[s]["macro_f1"]
                               for s in common]
            cond_deltas_common = [fp[s]["macro_f1"] - fa[s]["macro_f1"] for s in common]
            h2_diffs = [cond_deltas_common[i] - baseline_deltas[i] for i in range(len(common))]
            out["h2_paired_diffs"] = h2_diffs
            out["h2_mean"] = float(np.mean(h2_diffs))
            out["h2_wilcoxon_p"] = wilcoxon(h2_diffs)
            out["h2_seeds"] = common

    # Straggler-tolerance ratio (vs baseline)
    if baseline_fa is not None and baseline_fp is not None:
        baseline_seeds = sorted(set(baseline_fa) & set(baseline_fp))
        baseline_fa_mean = float(np.mean([baseline_fa[s]["macro_f1"] for s in baseline_seeds]))
        baseline_fp_mean = float(np.mean([baseline_fp[s]["macro_f1"] for s in baseline_seeds]))
        out["straggler_tolerance"] = {
            "fedavg":  out["fedavg_mean"]  / baseline_fa_mean,
            "fedprox": out["fedprox_mean"] / baseline_fp_mean,
        }

    return out


def main():
    print("=" * 72)
    print("SYSTEM HETEROGENEITY ANALYSIS")
    print("=" * 72)

    # Baseline (C0): headline statistical-het sweep
    print("\nLoading baseline (no system het)...")
    base_dir = RESULTS_ROOT / "headline"
    base_fa, base_fp = load_pairs(base_dir)
    print(f"  Loaded {len(base_fa)} FedAvg, {len(base_fp)} FedProx from {base_dir}")

    # C1 — fixed stragglers
    print("\nLoading C1 (fixed stragglers)...")
    c1_dir = RESULTS_ROOT / "system_het_fixed"
    if not c1_dir.exists():
        print(f"  Directory does not exist yet: {c1_dir}")
        c1_fa, c1_fp = {}, {}
    else:
        c1_fa, c1_fp = load_pairs(c1_dir)
        print(f"  Loaded {len(c1_fa)} FedAvg, {len(c1_fp)} FedProx from {c1_dir}")

    # C2 — random stragglers
    print("\nLoading C2 (random stragglers)...")
    c2_dir = RESULTS_ROOT / "system_het_random"
    if not c2_dir.exists():
        print(f"  Directory does not exist yet: {c2_dir}")
        c2_fa, c2_fp = {}, {}
    else:
        c2_fa, c2_fp = load_pairs(c2_dir)
        print(f"  Loaded {len(c2_fa)} FedAvg, {len(c2_fp)} FedProx from {c2_dir}")

    if not (c1_fa and c1_fp) and not (c2_fa and c2_fp):
        print("\nNo system-heterogeneity results found yet. Re-run when HPC sweeps complete.")
        return

    # Summarise each condition
    summaries = []
    summaries.append(summarise_condition(base_fa, base_fp, "C0 (baseline)"))
    if c1_fa and c1_fp:
        summaries.append(summarise_condition(c1_fa, c1_fp, "C1 (fixed_stragglers)",
                                              baseline_fa=base_fa, baseline_fp=base_fp))
    if c2_fa and c2_fp:
        summaries.append(summarise_condition(c2_fa, c2_fp, "C2 (random_stragglers)",
                                              baseline_fa=base_fa, baseline_fp=base_fp))

    # Print headline table
    print("\n" + "=" * 100)
    print("HEADLINE RESULTS TABLE")
    print("=" * 100)
    print(f"{'condition':<25} {'FedAvg':>14} {'FedProx':>14} {'Δ':>10} {'p (H1)':>8} {'p (H2)':>8} {'r_rb':>8}")
    print("-" * 100)
    for s in summaries:
        if s is None: continue
        p_h2 = s.get("h2_wilcoxon_p", float("nan"))
        print(f"{s['condition']:<25} "
              f"{s['fedavg_mean']:>7.4f}±{s['fedavg_sd']:.3f} "
              f"{s['fedprox_mean']:>7.4f}±{s['fedprox_sd']:.3f} "
              f"{s['delta_mean']:>+10.4f} "
              f"{s['wilcoxon_p_h1']:>8.4f} "
              f"{p_h2 if not np.isnan(p_h2) else float('nan'):>8.4f} "
              f"{s['rank_biserial']:>+8.3f}")

    # Straggler-tolerance ratios
    print("\nSTRAGGLER-TOLERANCE RATIOS (vs C0 baseline)")
    print("-" * 60)
    print(f"{'condition':<25} {'ρ_FedAvg':>12} {'ρ_FedProx':>12}")
    for s in summaries:
        if s is None or "straggler_tolerance" not in s:
            continue
        st = s["straggler_tolerance"]
        print(f"{s['condition']:<25} {st['fedavg']:>12.4f} {st['fedprox']:>12.4f}")

    # Save
    with open(DATA_DIR / "summary_statistics.json", "w") as f:
        json.dump({"conditions": summaries}, f, indent=2)
    print(f"\nWrote {DATA_DIR / 'summary_statistics.json'}")

    rows = []
    for s in summaries:
        if s is None: continue
        for i, seed in enumerate(s["seeds"]):
            rows.append({
                "condition": s["condition"],
                "seed": seed,
                "delta_macro_f1": s["per_seed_delta"][i],
            })
    pd.DataFrame(rows).to_csv(DATA_DIR / "per_seed_results.csv", index=False)
    print(f"Wrote {DATA_DIR / 'per_seed_results.csv'}")


if __name__ == "__main__":
    main()
