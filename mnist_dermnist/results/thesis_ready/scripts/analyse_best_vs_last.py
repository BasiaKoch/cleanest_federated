"""Best-vs-last round comparison from existing 20 history CSVs.

Inspired by Marija (2025) Figure 3.7 right panel: comparing each algorithm's
performance at its peak round vs at the final round reveals overfitting
dynamics. We compute this on validation macro-F1 (since test is only evaluated
once per run at best-val, never at the final round).

For each run we compute:
    val_macro_F1 at peak round (the round chosen for test evaluation)
    val_macro_F1 at round 150 (the final round)
    Drop = peak − final

Aggregating across 10 seeds gives the mean "post-peak drop" — a measure of how
much each algorithm overfits between the best and final round.

Reads:   mnist_dermnist/results/headline/history_*.csv
Writes:  thesis_ready/data/best_vs_last_round.csv
         thesis_ready/data/best_vs_last_round.json
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


ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
HEADLINE = ROOT.parent / "headline"


def main():
    rows = []
    pat = re.compile(r"history_(fedavg|fedprox)_mu([0-9.]+)_E20_s(\d+)\.csv")

    for f in sorted(HEADLINE.glob("history_*.csv")):
        m = pat.match(f.name)
        if not m: continue
        algo, mu, seed = m.group(1), float(m.group(2)), int(m.group(3))
        df = pd.read_csv(f)

        # peak round = argmax val_macro_f1
        idx = df["val_macro_f1"].idxmax()
        peak_round = int(df.loc[idx, "round"])
        peak_val   = float(df.loc[idx, "val_macro_f1"])

        # final round = round 150
        last_row = df[df["round"] == df["round"].max()].iloc[0]
        final_round = int(last_row["round"])
        final_val   = float(last_row["val_macro_f1"])

        drop = peak_val - final_val

        rows.append({
            "algorithm":   algo,
            "seed":        seed,
            "peak_round":  peak_round,
            "peak_val_macro_f1":  peak_val,
            "final_round": final_round,
            "final_val_macro_f1": final_val,
            "drop_peak_to_final": drop,
            "drop_pct_of_peak":   drop / peak_val if peak_val > 0 else 0,
        })

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "best_vs_last_per_seed.csv", index=False)

    # Summary stats per algorithm
    print(f"{'algorithm':>10} | {'mean peak F1':>12} | {'mean final F1':>13} | "
          f"{'mean drop':>10} | {'drop SD':>8}")
    print("-" * 70)
    summary = {}
    for algo in ["fedavg", "fedprox"]:
        sub = df[df.algorithm == algo]
        summary[algo] = {
            "n": len(sub),
            "mean_peak_round":     float(sub["peak_round"].mean()),
            "median_peak_round":   float(sub["peak_round"].median()),
            "mean_peak_val_f1":    float(sub["peak_val_macro_f1"].mean()),
            "sd_peak_val_f1":      float(sub["peak_val_macro_f1"].std(ddof=1)),
            "mean_final_val_f1":   float(sub["final_val_macro_f1"].mean()),
            "sd_final_val_f1":     float(sub["final_val_macro_f1"].std(ddof=1)),
            "mean_drop":           float(sub["drop_peak_to_final"].mean()),
            "sd_drop":             float(sub["drop_peak_to_final"].std(ddof=1)),
            "mean_drop_pct":       float(sub["drop_pct_of_peak"].mean()),
        }
        s = summary[algo]
        print(f"{algo:>10} | {s['mean_peak_val_f1']:>12.4f} | "
              f"{s['mean_final_val_f1']:>13.4f} | {s['mean_drop']:>10.4f} | "
              f"{s['sd_drop']:>8.4f}")

    # Paired test: is the drop different between algorithms?
    fa_drops = df[df.algorithm == "fedavg"]["drop_peak_to_final"].values
    fp_drops = df[df.algorithm == "fedprox"]["drop_peak_to_final"].values
    if HAS_SCIPY and len(fa_drops) == len(fp_drops):
        # Align by seed
        fa_by_seed = df[df.algorithm == "fedavg"].set_index("seed")["drop_peak_to_final"]
        fp_by_seed = df[df.algorithm == "fedprox"].set_index("seed")["drop_peak_to_final"]
        common = sorted(set(fa_by_seed.index) & set(fp_by_seed.index))
        paired_diffs = [fp_by_seed[s] - fa_by_seed[s] for s in common]
        try:
            _, p = stats.wilcoxon(paired_diffs, alternative="two-sided")
        except ValueError:
            p = float("nan")
        print(f"\nPaired Wilcoxon on (drop_FedProx − drop_FedAvg): p = {p:.4f}")
        print(f"  Mean paired diff = {np.mean(paired_diffs):+.4f}")
        summary["paired_drop_test"] = {
            "wilcoxon_p_two_sided": p,
            "mean_paired_diff":     float(np.mean(paired_diffs)),
            "interpretation":       "Negative mean = FedProx drops less than FedAvg "
                                     "(more stable post-peak)",
        }

    # Marija-style "best vs final" comparison table
    print(f"\n{'algorithm':>10} | {'best round F1':>14} | {'final round F1':>15}")
    print("-" * 50)
    for algo in ["fedavg", "fedprox"]:
        s = summary[algo]
        print(f"{algo:>10} | {s['mean_peak_val_f1']:>8.3f} ± {s['sd_peak_val_f1']:.3f} | "
              f"{s['mean_final_val_f1']:>8.3f} ± {s['sd_final_val_f1']:.3f}")

    with open(DATA_DIR / "best_vs_last_round.json", "w") as f:
        json.dump({
            "description": "Validation macro-F1 at peak round vs final round, "
                           "comparing overfitting dynamics between FedAvg and FedProx.",
            "n_seeds_per_algo": 10,
            "summary": summary,
        }, f, indent=2)
    print(f"\nWrote {DATA_DIR / 'best_vs_last_per_seed.csv'}")
    print(f"Wrote {DATA_DIR / 'best_vs_last_round.json'}")


if __name__ == "__main__":
    main()
