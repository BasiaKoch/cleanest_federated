"""Paired FedAvg vs FedProx analysis tables.

Outputs:
  - final_test_table.csv     : per-seed rows with FedAvg, FedProx, Δ
  - paired_stats.json        : mean±std, Wilcoxon p, rank-biserial r, n_pairs
  - per_class_diff.csv       : per-class F1 differences

Refuses to claim significance unless the paired test supports it. Prints
clear warnings when seeds are missing or pairs are incomplete.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps


CLASS_NAMES = (
    "actinic_keratoses",
    "basal_cell_carcinoma",
    "benign_keratosis_like_lesions",
    "dermatofibroma",
    "melanoma",
    "melanocytic_nevi",
    "vascular_lesions",
)

# Filename schema (see fl/server_loop.py:save_run_outputs and the Flower runners):
#   test_at_best_<algo>_mu<mu>_E<E>[_sh-<sh_mode>][_C<frac>]_s<seed>.json
# where:
#   - <algo>     ∈ {fedavg, fedprox, fednova}
#   - _sh-<...>  is present only for non-uniform system-het modes
#   - _C<frac>   is present only for fraction_fit ≠ 1.0
# A regex that requires `_s<seed>` immediately after `E<E>` (an earlier
# version of this file) silently skipped every system-het and FedNova
# JSON, so paired_compare returned empty results without any error.
_PAT = re.compile(
    r"test_at_best_"
    r"(?P<algo>fedavg|fedprox|fednova)"
    r"_mu(?P<mu>[0-9.]+)"
    r"_E(?P<E>\d+)"
    r"(?:_sh-(?P<sh>[a-z_]+))?"
    r"(?:_C(?P<C>[0-9.]+))?"
    r"_s(?P<seed>\d+)\.json"
)


def _discover(results_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(results_dir.glob("test_at_best_*.json")):
        m = _PAT.match(f.name)
        if not m:
            warnings.warn(f"Ignoring unrecognized filename: {f.name}")
            continue
        with open(f) as fp:
            data = json.load(fp)
        rows.append({
            "algorithm": m["algo"],
            "mu": float(m["mu"]),
            "local_epochs": int(m["E"]),
            "sh_mode": m["sh"] or "uniform",
            "fraction_fit": float(m["C"]) if m["C"] else 1.0,
            "seed": int(m["seed"]),
            "test_macro_f1": float(data["macro_f1"]),
            "test_balanced_accuracy": float(data["balanced_accuracy"]),
            "test_accuracy": float(data["accuracy"]),
            "test_loss": float(data["loss"]),
            "selected_round": int(data.get("selected_round", -1)),
            "best_val_macro_f1": float(data.get("best_val_macro_f1", float("nan"))),
            "per_class_f1": data.get("per_class_f1", [float("nan")] * 7),
            "file": f.name,
        })
    return pd.DataFrame(rows)


def _rank_biserial(diffs: np.ndarray) -> float:
    """Effect size for paired signed-rank — interpretable on [-1, 1]."""
    d = diffs[diffs != 0]
    if len(d) == 0:
        return float("nan")
    ranks = sps.rankdata(np.abs(d))
    w_pos = float(ranks[d > 0].sum())
    w_neg = float(ranks[d < 0].sum())
    return (w_pos - w_neg) / (w_pos + w_neg)


def paired_compare(df: pd.DataFrame, E: int = 20) -> Dict:
    """Build the seed-aligned paired comparison at a given E."""
    sub = df[df["local_epochs"] == E]
    if sub.empty:
        raise ValueError(f"No runs found for E={E}")

    fa = sub[sub["algorithm"] == "fedavg"].set_index("seed")
    fp = sub[sub["algorithm"] == "fedprox"].set_index("seed")

    seeds_fa = set(fa.index)
    seeds_fp = set(fp.index)
    paired = sorted(seeds_fa & seeds_fp)
    missing_fa = sorted(seeds_fp - seeds_fa)
    missing_fp = sorted(seeds_fa - seeds_fp)

    if missing_fa:
        warnings.warn(f"FedAvg missing for seeds {missing_fa} (E={E}). Those seeds are excluded.")
    if missing_fp:
        warnings.warn(f"FedProx missing for seeds {missing_fp} (E={E}). Those seeds are excluded.")
    if len(paired) < 2:
        warnings.warn(f"Only {len(paired)} paired seed(s) available. Statistical tests are unreliable.")

    if not paired:
        return {"E": E, "n_pairs": 0, "error": "no paired seeds"}

    fa_val = fa.loc[paired, "test_macro_f1"].to_numpy(dtype=float)
    fp_val = fp.loc[paired, "test_macro_f1"].to_numpy(dtype=float)
    diff = fp_val - fa_val

    # Tests
    try:
        w_stat, w_p2 = sps.wilcoxon(fp_val, fa_val, alternative="two-sided")
    except ValueError:
        w_stat, w_p2 = float("nan"), float("nan")
    try:
        _, w_p_greater = sps.wilcoxon(fp_val, fa_val, alternative="greater")
    except ValueError:
        w_p_greater = float("nan")
    sig2 = ("***" if w_p2 < 0.001
            else "**" if w_p2 < 0.01
            else "*"  if w_p2 < 0.05
            else "ns")

    return {
        "E": E,
        "n_pairs": len(paired),
        "paired_seeds": paired,
        "missing_fedavg_seeds": missing_fa,
        "missing_fedprox_seeds": missing_fp,
        "fedavg_mean": float(fa_val.mean()),
        "fedavg_std": float(fa_val.std(ddof=1)) if len(fa_val) > 1 else 0.0,
        "fedprox_mean": float(fp_val.mean()),
        "fedprox_std": float(fp_val.std(ddof=1)) if len(fp_val) > 1 else 0.0,
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
        "diffs_per_seed": [float(d) for d in diff],
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p_two_sided": float(w_p2),
        "wilcoxon_p_greater": float(w_p_greater),
        "wilcoxon_significance": sig2,
        "rank_biserial": _rank_biserial(diff),
    }


def per_class_diff_table(df: pd.DataFrame, E: int = 20) -> pd.DataFrame:
    """Class-by-class FedProx − FedAvg, averaged across paired seeds."""
    sub = df[df["local_epochs"] == E]
    fa = sub[sub["algorithm"] == "fedavg"].set_index("seed")
    fp = sub[sub["algorithm"] == "fedprox"].set_index("seed")
    paired = sorted(set(fa.index) & set(fp.index))
    if not paired:
        return pd.DataFrame()
    rows = []
    for c in range(7):
        fa_v = np.array([fa.loc[s, "per_class_f1"][c] for s in paired])
        fp_v = np.array([fp.loc[s, "per_class_f1"][c] for s in paired])
        d = fp_v - fa_v
        try:
            _, p = sps.wilcoxon(fp_v, fa_v, alternative="two-sided") if len(d) >= 2 else (float("nan"), float("nan"))
        except ValueError:
            p = float("nan")
        rows.append({
            "class_id": c,
            "class_name": CLASS_NAMES[c],
            "fedavg_mean": float(fa_v.mean()),
            "fedprox_mean": float(fp_v.mean()),
            "mean_diff": float(d.mean()),
            "n_pairs": len(d),
            "wilcoxon_p": float(p),
        })
    return pd.DataFrame(rows)


def format_report(stats: Dict) -> str:
    if stats.get("error"):
        return f"ERROR: {stats['error']}"
    lines = []
    lines.append("=" * 70)
    lines.append(f"Paired FedAvg vs FedProx — E={stats['E']}, n_pairs={stats['n_pairs']}")
    lines.append(f"Paired seeds: {stats['paired_seeds']}")
    if stats["missing_fedavg_seeds"]:
        lines.append(f"WARN: FedAvg missing for seeds {stats['missing_fedavg_seeds']}")
    if stats["missing_fedprox_seeds"]:
        lines.append(f"WARN: FedProx missing for seeds {stats['missing_fedprox_seeds']}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  FedAvg  test macro-F1 : {stats['fedavg_mean']:.4f} ± {stats['fedavg_std']:.4f}")
    lines.append(f"  FedProx test macro-F1 : {stats['fedprox_mean']:.4f} ± {stats['fedprox_std']:.4f}")
    lines.append(f"  Mean paired diff      : {stats['mean_diff']:+.4f} ± {stats['std_diff']:.4f}")
    lines.append(f"  Per-seed diffs        : {[f'{d:+.4f}' for d in stats['diffs_per_seed']]}")
    lines.append("")
    lines.append(f"  Wilcoxon two-sided p  : {stats['wilcoxon_p_two_sided']:.4f}  [{stats['wilcoxon_significance']}]")
    lines.append(f"  Wilcoxon greater  p   : {stats['wilcoxon_p_greater']:.4f}  (H1: FedProx > FedAvg)")
    lines.append(f"  Rank-biserial effect  : {stats['rank_biserial']:+.3f}")
    lines.append("")
    # Verdict
    if stats["n_pairs"] < 5:
        lines.append("VERDICT: insufficient seeds for reliable inference; report as descriptive only.")
    elif stats["wilcoxon_p_two_sided"] < 0.05 and stats["mean_diff"] > 0:
        lines.append("VERDICT: FedProx significantly outperforms FedAvg (Wilcoxon p<0.05).")
    elif stats["wilcoxon_p_two_sided"] < 0.05 and stats["mean_diff"] < 0:
        lines.append("VERDICT: FedAvg significantly outperforms FedProx (Wilcoxon p<0.05).")
    else:
        lines.append("VERDICT: no statistically significant difference at α=0.05.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="mnist_dermnist/results/headline")
    ap.add_argument("--E", type=int, default=20)
    ap.add_argument("--out", default=None,
                    help="Output dir (default: <results-dir>/analysis/)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"results-dir not found: {results_dir}")
    out_dir = Path(args.out) if args.out else (results_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _discover(results_dir)
    if df.empty:
        raise SystemExit(f"No test_at_best_*.json files found in {results_dir}")

    # Per-seed table at this E
    sub = df[df["local_epochs"] == args.E].copy()
    sub_table = sub.pivot_table(index="seed", columns="algorithm", values="test_macro_f1", aggfunc="first")
    if "fedavg" in sub_table.columns and "fedprox" in sub_table.columns:
        sub_table["diff"] = sub_table["fedprox"] - sub_table["fedavg"]
    sub_table.to_csv(out_dir / "final_test_table.csv")
    print(f"\nWrote {out_dir / 'final_test_table.csv'}")
    print(sub_table.to_string())

    # Paired stats
    s = paired_compare(df, E=args.E)
    print()
    print(format_report(s))
    with open(out_dir / "paired_stats.json", "w") as f:
        json.dump(s, f, indent=2)
    print(f"\nWrote {out_dir / 'paired_stats.json'}")

    # Per-class diff
    pc = per_class_diff_table(df, E=args.E)
    if not pc.empty:
        pc.to_csv(out_dir / "per_class_diff.csv", index=False)
        print(f"\nPer-class F1 differences (FedProx − FedAvg):")
        print(pc.to_string(index=False))


if __name__ == "__main__":
    main()
