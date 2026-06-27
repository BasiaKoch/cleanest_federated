"""Update-norm mechanism evidence figure -- 2-panel: trajectory + scatter.

Panel A (engineered trajectory):
  Reads per-client per-round update-norm CSVs from the engineered Flower sweep:
    results/flower_C0_baseline/   (engineered partition)
  Mean (across clients within each round) per-round update-norm trajectory
  for FedAvg vs FedProx, averaged across paired seeds with +/-SEM band.
  FedProx tracks lower throughout (round-mean ~1.029 vs ~1.523, -32%).

Panel B (cross-protocol scatter):
  Reads the pooled L4 cross-protocol table:
    results/thesis_ready/data/l4_cross_protocol_mechanism.csv  (78 runs, 8 cells)
  Dominant-client (Client 0) mean update norm vs rare-class F1, one point per
  run, coloured by experiment cell. Spearman r = +0.634 (dominant-client norm
  vs rare-F1); the relative-influence ratio (Client1/Client0) is r = -0.045.

Output:
  results/thesis_ready/figures/F_update_norms.{pdf,png}

Style: matches F_iid_convergence, F_engineered_convergence,
F_partition_robustness -- axes, legend, panel subtitle only; no
figure title; no inset notes.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


THIS = Path(__file__).resolve()
from fl_dermamnist.common.paths import repo_root  # noqa: E402
REPO_ROOT = repo_root()
RESULTS_ROOT = REPO_ROOT / "fl_dermamnist" / "results"
OUT_FIG = RESULTS_ROOT / "thesis_ready" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

SCATTER_CSV = RESULTS_ROOT / "thesis_ready" / "data" / "l4_cross_protocol_mechanism.csv"

NUM_ROUNDS = 150

COL_FEDAVG  = "#7FBF94"
COL_FEDPROX = "#3D5A80"


plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":         11,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})


def load_per_round(sweep_dir: Path, algo: str, mu: str) -> dict[int, np.ndarray]:
    """Per-seed array of length NUM_ROUNDS: mean update norm across clients per round."""
    out: dict[int, np.ndarray] = {}
    pat = re.compile(rf"client_update_norms_{algo}_mu{mu}_E20_s(\d+)\.csv")
    for f in sorted(sweep_dir.glob(f"client_update_norms_{algo}_*.csv")):
        m = pat.match(f.name)
        if not m: continue
        seed = int(m.group(1))
        df = pd.read_csv(f)
        per_round = df.groupby("round")["update_norm"].mean()
        arr = np.full(NUM_ROUNDS, np.nan)
        for r, v in per_round.items():
            if 1 <= r <= NUM_ROUNDS:
                arr[r - 1] = float(v)
        out[seed] = arr
    return out


def stack(curves: dict[int, np.ndarray]):
    rounds = np.arange(1, NUM_ROUNDS + 1)
    if not curves:
        return rounds, np.full(NUM_ROUNDS, np.nan), np.full(NUM_ROUNDS, np.nan)
    mat = np.stack(list(curves.values()))
    mean = np.nanmean(mat, axis=0)
    sem  = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(mat), axis=0))
    return rounds, mean, sem


def _traj_panel(ax, fa, fp, show_legend=False):
    for curves, col, lbl in [
        (fa, COL_FEDAVG,  "FedAvg"),
        (fp, COL_FEDPROX, "FedProx ($\\mu = 0.01$)"),
    ]:
        rounds, mean, sem = stack(curves)
        ax.plot(rounds, mean, color=col, linewidth=1.6, label=lbl)
        ax.fill_between(rounds, mean - sem, mean + sem,
                        color=col, alpha=0.20, linewidth=0)
    ax.set_xlim(1, NUM_ROUNDS)
    ax.set_ylim(0, 4.2)
    ax.set_xlabel("Communication round")
    ax.set_ylabel(r"Mean per-client update norm $\|w_i^{t+1} - w^t\|_2$")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    if show_legend:
        ax.legend(loc="upper right", frameon=False)


def _scatter_panel(ax):
    df = pd.read_csv(SCATTER_CSV)
    valid = df.dropna(subset=["mean_norm_c0", "rare_avg_f1",
                              "influence_ratio_c1_to_c0"])
    # Spearman correlations quoted in the caption.
    r_norm, _  = spearmanr(valid["mean_norm_c0"], valid["rare_avg_f1"])
    r_ratio, _ = spearmanr(valid["influence_ratio_c1_to_c0"], valid["rare_avg_f1"])
    print(f"Scatter: n={len(valid)}  "
          f"Spearman(dominant-norm, rare-F1)={r_norm:+.3f}  "
          f"Spearman(influence-ratio, rare-F1)={r_ratio:+.3f}")

    # One colour per experiment cell, taken from the table's own colour column.
    seen: dict[str, str] = {}
    for lbl, col in zip(valid["experiment_label"], valid["experiment_color"]):
        seen.setdefault(lbl, col)
    for lbl, col in seen.items():
        sub = valid[valid["experiment_label"] == lbl]
        ax.scatter(sub["mean_norm_c0"], sub["rare_avg_f1"],
                   s=34, color=col, edgecolor="white", linewidth=0.4,
                   alpha=0.9, label=lbl)

    ax.set_xlabel(r"Dominant-client mean update norm $\overline{\|\Delta w_0\|}_2$")
    ax.set_ylabel("Rare-class F1")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=-0.02)

    ax.text(0.04, 0.96,
            f"Spearman $r = {r_norm:+.3f}$\n"
            f"(influence ratio $r = {r_ratio:+.3f}$)",
            transform=ax.transAxes, ha="left", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="0.7", alpha=0.9))
    ax.legend(loc="lower right", frameon=False, fontsize=7.2,
              handletextpad=0.2, labelspacing=0.25)


def main():
    eng_fa = load_per_round(RESULTS_ROOT / "flower_C0_baseline", "fedavg",  "0.0")
    eng_fp = load_per_round(RESULTS_ROOT / "flower_C0_baseline", "fedprox", "0.01")
    print(f"Engineered: n_fa={len(eng_fa)}, n_fp={len(eng_fp)}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                             gridspec_kw={"wspace": 0.22})
    _traj_panel(axes[0], eng_fa, eng_fp, show_legend=True)
    _scatter_panel(axes[1])

    axes[0].text(0.5, -0.20, "Engineered partition (non-IID)",
                 transform=axes[0].transAxes, ha="center", fontsize=11)
    axes[1].text(0.5, -0.20, "Cross-protocol pool (78 L4 runs)",
                 transform=axes[1].transAxes, ha="center", fontsize=11)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_FIG / f"F_update_norms.{ext}"
        fig.savefig(out)
        print(f"Wrote: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
