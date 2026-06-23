"""Per-class convergence on the Flower runtime -- F8.

Sibling of F2 (pure-PyTorch per-class convergence). Reads
results/flower_C0_baseline/history_*.csv (the runtime-matched headline
re-run, n=10 paired seeds, engineered partition, no system het) and
plots a 2x4 grid: one panel per class showing val_f1_class_<c> over
rounds for FedAvg vs FedProx, mean +/- SEM across the ten seeds.

This figure characterises learning dynamics on the validation set only.
Test-set per-class deltas, p-values, and significance markers are
reported in Table T02 (per-class Holm-corrected paired Wilcoxon).

Output:
  results/thesis_ready/figures/F8_per_class_convergence_flower.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS = Path(__file__).resolve()
from fl_dermamnist.common.paths import repo_root  # noqa: E402
REPO_ROOT = repo_root()
RESULTS_ROOT = REPO_ROOT / "fl_dermamnist" / "results"
SWEEP = RESULTS_ROOT / "flower_C0_baseline"
OUT_FIG = RESULTS_ROOT / "thesis_ready" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

NUM_ROUNDS = 150
PAIRED_SEEDS = [42, 123, 456, 789, 999, 2024, 31337, 161803, 271828, 8675309]

CLASS_DISPLAY = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevi",
    "Vascular lesions",
]

COL_FEDAVG  = "#7FBF94"   # soft mint green
COL_FEDPROX = "#3D5A80"   # dark slate blue


plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":   11,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})


def load_history(sweep_dir: Path, algo: str, mu: str) -> pd.DataFrame:
    frames = []
    for seed in PAIRED_SEEDS:
        p = sweep_dir / f"history_{algo}_mu{mu}_E20_s{seed}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        if "seed" not in df.columns:
            df["seed"] = seed
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No history files in {sweep_dir} for {algo} mu={mu}")
    return pd.concat(frames, ignore_index=True)


def mean_sem_curve(df: pd.DataFrame, ycol: str):
    g = df.groupby("round")[ycol]
    rounds = np.array(sorted(df["round"].unique()))
    m = g.mean().reindex(rounds).to_numpy()
    n = g.count().reindex(rounds).to_numpy()
    sd = g.std().reindex(rounds).to_numpy()
    sem = sd / np.sqrt(np.clip(n, 1, None))
    return rounds, m, sem


def plot_curve(ax, rounds, mean, sem, *, color, label):
    ax.plot(rounds, mean, color=color, label=label, linewidth=1.5)
    ax.fill_between(rounds, mean - sem, mean + sem,
                    color=color, alpha=0.15, linewidth=0)


def main():
    print(f"Reading Flower headline history from {SWEEP}")
    fa_df = load_history(SWEEP, "fedavg",  "0.0")
    fp_df = load_history(SWEEP, "fedprox", "0.01")
    print(f"  FedAvg:  {fa_df['seed'].nunique()} seeds")
    print(f"  FedProx: {fp_df['seed'].nunique()} seeds")

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    axes = axes.flatten()

    for c in range(7):
        ax = axes[c]
        ycol = f"val_f1_class_{c}"
        r1, fa_m, fa_s = mean_sem_curve(fa_df, ycol)
        r2, fp_m, fp_s = mean_sem_curve(fp_df, ycol)
        plot_curve(ax, r1, fa_m, fa_s, color=COL_FEDAVG,  label="FedAvg")
        plot_curve(ax, r2, fp_m, fp_s, color=COL_FEDPROX, label="FedProx ($\\mu=0.01$)")
        ax.set_title(CLASS_DISPLAY[c], loc="left", fontweight="bold", pad=6, fontsize=10)
        ax.set_xlabel("Communication round")
        ax.set_ylabel("Validation F1")
        ax.set_xlim(0, NUM_ROUNDS)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
        if c == 0:
            ax.legend(loc="upper left", frameon=False, fontsize=9)

    # 8th cell -> notes panel (validation-curve description only;
    # test-set deltas are reported in Table T02, not here).
    ax = axes[7]
    ax.axis("off")
    ax.text(0.05, 0.95, "Notes:", fontweight="bold",
            transform=ax.transAxes, va="top")
    notes = [
        r"Flower runtime, $n=10$",
        r"paired seeds, engineered",
        r"partition, no system het.",
        "",
        r"Curves: mean $\pm$ SEM of",
        r"val\_f1\_class\_$c$ over rounds.",
        "",
        r"Validation set only;",
        r"test-set per-class deltas",
        r"and Holm-corrected $p$-values",
        r"are in Table T02.",
    ]
    for i, t in enumerate(notes):
        ax.text(0.05, 0.88 - 0.06 * i, t,
                transform=ax.transAxes, va="top", fontsize=9)

    fig.suptitle(
        "Per-class validation F1 over rounds "
        "(Flower runtime, $n=10$ paired seeds, mean $\\pm$ SEM)",
        fontsize=11, y=1.01, fontweight="bold"
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_FIG / f"F8_per_class_convergence_flower.{ext}"
        fig.savefig(out)
        print(f"Wrote: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
