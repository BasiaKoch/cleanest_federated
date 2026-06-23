"""Update-norm mechanism evidence figure -- clean 2-panel cross-partition.

Reads per-client per-round update-norm CSVs from two Flower sweeps:
  results/flower_C0_baseline/      (engineered partition)
  results/flower_C0_iid_baseline/  (IID partition)

Each panel: mean (across clients within each round) per-round update
norm trajectory for FedAvg vs FedProx, averaged across paired seeds
with +/-SEM band.

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


THIS = Path(__file__).resolve()
from fl_dermamnist.common.paths import repo_root  # noqa: E402
REPO_ROOT = repo_root()
RESULTS_ROOT = REPO_ROOT / "fl_dermamnist" / "results"
OUT_FIG = RESULTS_ROOT / "thesis_ready" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

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


def _panel(ax, fa, fp, show_legend=False):
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
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    if show_legend:
        ax.legend(loc="upper right", frameon=False)


def main():
    eng_fa = load_per_round(RESULTS_ROOT / "flower_C0_baseline",     "fedavg",  "0.0")
    eng_fp = load_per_round(RESULTS_ROOT / "flower_C0_baseline",     "fedprox", "0.01")
    iid_fa = load_per_round(RESULTS_ROOT / "flower_C0_iid_baseline", "fedavg",  "0.0")
    iid_fp = load_per_round(RESULTS_ROOT / "flower_C0_iid_baseline", "fedprox", "0.01")
    print(f"Engineered: n_fa={len(eng_fa)}, n_fp={len(eng_fp)}")
    print(f"IID:        n_fa={len(iid_fa)}, n_fp={len(iid_fp)}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                             sharey=True,
                             gridspec_kw={"wspace": 0.10})
    _panel(axes[0], eng_fa, eng_fp, show_legend=True)
    axes[0].set_ylabel(r"Mean per-client update norm $\|w_i^{t+1} - w^t\|_2$")
    _panel(axes[1], iid_fa, iid_fp, show_legend=False)

    axes[0].text(0.5, -0.20, "Engineered partition (non-IID)",
                 transform=axes[0].transAxes, ha="center", fontsize=11)
    axes[1].text(0.5, -0.20, "IID partition (mechanism null)",
                 transform=axes[1].transAxes, ha="center", fontsize=11)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_FIG / f"F_update_norms.{ext}"
        fig.savefig(out)
        print(f"Wrote: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
