"""Analyse the system-heterogeneity HPC results.

Reads `test_at_best_*.json` files from
  - mnist_dermnist/results/system_het_fixed/        (C1: Flower)
  - mnist_dermnist/results/system_het_random/       (C2: Flower)
and aggregates them into per-condition paired test statistics (H1) plus
between-condition tests of whether system heterogeneity amplifies the
FedProx advantage (H2).

H2 BASELINE — RUNTIME-MATCHED
-----------------------------
The H2 contrast subtracts the C0 per-seed Δ from each condition's Δ.
C0 MUST come from the SAME runtime as C1/C2, otherwise the H2 number
partially reflects pure-PyTorch ↔ Flower equivalence noise rather than
the system-heterogeneity manipulation itself.

Earlier versions of this script read `mnist_dermnist/results/headline/`
as the C0 baseline; those files are from the PURE-PYTORCH reference
loop, while C1/C2 are Flower outputs. That mixing has been removed.
This script now requires `mnist_dermnist/results/flower_C0_baseline/`
(produced by `submit_flower_C0_baseline.sh`) and refuses to compute H2
if either (a) that directory does not exist, or (b) any baseline JSON
reports a `framework` other than `flower-simulation`.

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


def load_pairs(results_dir: Path, *, require_framework: str | None = None):
    """Return (fedavg_by_seed, fedprox_by_seed, fednova_by_seed) dicts.

    If `require_framework` is given, raises ValueError when any loaded
    JSON's `framework` field disagrees. Used to enforce runtime-matched
    C0 for the H2 contrast.
    """
    fa, fp, fn = {}, {}, {}
    pat = re.compile(
        r"test_at_best_(fedavg|fedprox|fednova)_mu[0-9.]+_E20"
        r"(?:_sh-[a-z_]+)?(?:_C[0-9.]+)?_s(\d+)\.json"
    )
    framework_violations = []
    for f in sorted(results_dir.glob("test_at_best_*.json")):
        m = pat.match(f.name)
        if not m:
            print(f"  skipped (filename mismatch): {f.name}")
            continue
        algo, seed = m.group(1), int(m.group(2))
        data = json.load(open(f))
        if require_framework is not None:
            fw = data.get("framework", "<missing>")
            if fw != require_framework:
                framework_violations.append((f.name, fw))
                continue
        if algo == "fedavg":   fa[seed] = data
        elif algo == "fedprox": fp[seed] = data
        elif algo == "fednova": fn[seed] = data
    if framework_violations:
        raise ValueError(
            f"Refusing to use {results_dir.name} as C0 baseline: "
            f"{len(framework_violations)} file(s) report framework != "
            f"{require_framework!r}. First few: {framework_violations[:3]}. "
            f"H2 requires same-runtime C0; re-run with the correct submission "
            f"script (submit_flower_C0_baseline.sh) before retrying."
        )
    return fa, fp, fn


def summarise_solo(arm: dict, name: str) -> dict | None:
    """Per-arm summary for an algorithm that has no paired comparator
    in this script (FedNova is reported alongside FedAvg/FedProx but
    its statistical inference is unpaired here)."""
    seeds = sorted(arm)
    if not seeds:
        return None
    vals = [arm[s]["macro_f1"] for s in seeds]
    return {
        "algorithm": name,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "mean": float(np.mean(vals)),
        "sd":   float(np.std(vals, ddof=1)) if len(seeds) > 1 else 0.0,
        "per_seed": vals,
    }


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

    # Baseline (C0): RUNTIME-MATCHED Flower C0 sweep.
    # See module docstring — cross-runtime C0 (e.g. pure-PyTorch headline
    # vs Flower C1/C2) would confound the H2 inference with runtime
    # equivalence noise, so we refuse that combination.
    print("\nLoading baseline (no system het, Flower runtime)...")
    base_dir = RESULTS_ROOT / "flower_C0_baseline"
    if not base_dir.exists():
        print(f"  ERROR: {base_dir} does not exist.")
        print( "         H2 requires a Flower-runtime C0 baseline.")
        print( "         Run mnist_dermnist/scripts/submit_flower_C0_baseline.sh on HPC,")
        print( "         wait for it to complete, then re-run this analysis.")
        print( "         (The previous design used the pure-PyTorch headline as C0,")
        print( "          which would mix runtimes and invalidate H2; that path is")
        print( "          now removed.)")
        return
    base_fa, base_fp, base_fn = load_pairs(base_dir, require_framework="flower-simulation")
    print(f"  Loaded {len(base_fa)} FedAvg, {len(base_fp)} FedProx, "
          f"{len(base_fn)} FedNova from {base_dir}")
    if not (base_fa and base_fp):
        print(f"  ERROR: no paired FedAvg/FedProx baseline runs found in {base_dir}.")
        return

    # C1 — fixed stragglers
    print("\nLoading C1 (fixed stragglers)...")
    c1_dir = RESULTS_ROOT / "system_het_fixed"
    if not c1_dir.exists():
        print(f"  Directory does not exist yet: {c1_dir}")
        c1_fa, c1_fp, c1_fn = {}, {}, {}
    else:
        c1_fa, c1_fp, c1_fn = load_pairs(c1_dir)
        print(f"  Loaded {len(c1_fa)} FedAvg, {len(c1_fp)} FedProx, "
              f"{len(c1_fn)} FedNova from {c1_dir}")

    # C2 — random stragglers (FedAvg/FedProx in system_het_random;
    # FedNova in system_het_random_fednova per submit_fednova_system_het.sh)
    print("\nLoading C2 (random stragglers)...")
    c2_dir = RESULTS_ROOT / "system_het_random"
    c2_fn_dir = RESULTS_ROOT / "system_het_random_fednova"
    if not c2_dir.exists():
        print(f"  Directory does not exist yet: {c2_dir}")
        c2_fa, c2_fp, _ = {}, {}, {}
    else:
        c2_fa, c2_fp, _ = load_pairs(c2_dir)
        print(f"  Loaded {len(c2_fa)} FedAvg, {len(c2_fp)} FedProx from {c2_dir}")
    if c2_fn_dir.exists():
        _, _, c2_fn = load_pairs(c2_fn_dir)
        print(f"  Loaded {len(c2_fn)} FedNova from {c2_fn_dir}")
    else:
        c2_fn = {}
        print(f"  FedNova C2 directory does not exist yet: {c2_fn_dir}")

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

    # FedNova arms — reported alongside but not as a paired contrast
    # against FedAvg/FedProx (FedNova has its own objective). For each
    # condition where FedNova ran, we report mean+SD across seeds and
    # — where the same seed has all three algorithms — within-seed
    # differences FedNova - FedAvg and FedNova - FedProx.
    fednova_arms = []
    for cond_name, fn_dict, fa_dict, fp_dict in [
        ("C0 (baseline)",       base_fn, base_fa, base_fp),
        ("C1 (fixed_stragglers)", c1_fn,  c1_fa,  c1_fp),
        ("C2 (random_stragglers)", c2_fn, c2_fa,  c2_fp),
    ]:
        solo = summarise_solo(fn_dict, "FedNova")
        if solo is None:
            continue
        # Triple-paired Δ: only seeds present in all three arms
        triple_seeds = sorted(set(fn_dict) & set(fa_dict) & set(fp_dict))
        if triple_seeds:
            d_nova_avg = [fn_dict[s]["macro_f1"] - fa_dict[s]["macro_f1"]
                          for s in triple_seeds]
            d_nova_prox = [fn_dict[s]["macro_f1"] - fp_dict[s]["macro_f1"]
                           for s in triple_seeds]
            solo["delta_vs_fedavg_mean"]  = float(np.mean(d_nova_avg))
            solo["delta_vs_fedprox_mean"] = float(np.mean(d_nova_prox))
            solo["delta_vs_fedavg_p"]  = wilcoxon(d_nova_avg)
            solo["delta_vs_fedprox_p"] = wilcoxon(d_nova_prox)
            solo["triple_seeds"] = triple_seeds
        solo["condition"] = cond_name
        fednova_arms.append(solo)

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

    # FedNova arms (HV1) — reported as a third comparator
    if fednova_arms:
        print("\n" + "=" * 100)
        print("FEDNOVA ARM (per condition; triple-paired Δ vs FedAvg / vs FedProx where available)")
        print("=" * 100)
        print(f"{'condition':<25} {'FedNova':>14} {'n':>4} {'Δ vs FedAvg':>14} {'p':>8} {'Δ vs FedProx':>14} {'p':>8}")
        print("-" * 100)
        for a in fednova_arms:
            dva   = a.get("delta_vs_fedavg_mean",  float("nan"))
            dvp   = a.get("delta_vs_fedprox_mean", float("nan"))
            pva   = a.get("delta_vs_fedavg_p",     float("nan"))
            pvp   = a.get("delta_vs_fedprox_p",    float("nan"))
            print(f"{a['condition']:<25} "
                  f"{a['mean']:>7.4f}±{a['sd']:.3f} "
                  f"{a['n_seeds']:>4d} "
                  f"{dva:>+14.4f} {pva:>8.4f} "
                  f"{dvp:>+14.4f} {pvp:>8.4f}")

    # Save
    with open(DATA_DIR / "summary_statistics.json", "w") as f:
        json.dump({"conditions": summaries, "fednova_arms": fednova_arms}, f, indent=2)
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
