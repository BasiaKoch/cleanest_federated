"""System-heterogeneity convergence curves -- F7.

Three-panel layout showing FedAvg-vs-FedProx validation macro-F1
trajectories under each of the three system-heterogeneity conditions
(C0 baseline, C1 fixed stragglers, C2 random stragglers). All curves
are mean +/- standard error of the mean across the ten paired seeds.

Auxiliary fourth panel: FedNova C0 vs FedNova C2, illustrating the
catastrophic collapse of normalised-averaging under random stragglers.

Output:
  results/thesis_ready/figures/F7_system_het_convergence.{pdf,png}
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[4]
RESULTS_ROOT = REPO_ROOT / "mnist_dermnist" / "results"
OUT_FIG = REPO_ROOT / "mnist_dermnist" / "results" / "thesis_ready" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

NUM_ROUNDS = 150

COLOURS = {
    "fedavg":  "#7FBF94",   # soft mint green
    "fedprox": "#3D5A80",   # dark slate blue
    "fednova": "#9C6644",   # warm brown
}
LABEL = {
    "fedavg":  "FedAvg",
    "fedprox": "FedProx ($\\mu = 0.01$)",
    "fednova": "FedNova",
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":   11,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})


def _read_history_dir(directory: Path, algo: str, mu: str) -> dict[int, pd.DataFrame]:
    """Return {seed: dataframe with round + val_macro_f1} for the given (dir, algo)."""
    out: dict[int, pd.DataFrame] = {}
    pat = re.compile(rf"history_{algo}_mu{mu}_E20"
                     rf"(?:_sh-[a-z_]+)?_s(\d+)\.csv")
    for f in sorted(directory.glob(f"history_{algo}_*.csv")):
        m = pat.match(f.name)
        if not m:
            continue
        seed = int(m.group(1))
        df = pd.read_csv(f)
        df = df[["round", "val_macro_f1"]].copy()
        out[seed] = df
    return out


def _stack(curves: dict[int, pd.DataFrame]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rounds, mean, sem) across the stacked curves."""
    if not curves:
        return np.array([]), np.array([]), np.array([])
    rounds = np.arange(1, NUM_ROUNDS + 1)
    mat = np.full((len(curves), NUM_ROUNDS), np.nan)
    for i, (seed, df) in enumerate(curves.items()):
        for _, row in df.iterrows():
            r = int(row["round"])
            if 1 <= r <= NUM_ROUNDS:
                mat[i, r - 1] = float(row["val_macro_f1"])
    mean = np.nanmean(mat, axis=0)
    sem = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(mat), axis=0))
    return rounds, mean, sem


def _plot_condition(ax, fa_curves, fp_curves, title, *, fn_curves=None):
    for algo, curves in [("fedavg", fa_curves), ("fedprox", fp_curves)]:
        if not curves:
            continue
        rounds, mean, sem = _stack(curves)
        ax.plot(rounds, mean, color=COLOURS[algo], linewidth=1.6,
                label=LABEL[algo])
        ax.fill_between(rounds, mean - sem, mean + sem,
                        color=COLOURS[algo], alpha=0.18, linewidth=0)
    if fn_curves:
        rounds, mean, sem = _stack(fn_curves)
        ax.plot(rounds, mean, color=COLOURS["fednova"], linewidth=1.6,
                linestyle="--", label=LABEL["fednova"])
        ax.fill_between(rounds, mean - sem, mean + sem,
                        color=COLOURS["fednova"], alpha=0.15, linewidth=0)
    ax.set_xlim(1, NUM_ROUNDS)
    ax.set_ylim(0.0, 0.62)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title(title, loc="left", fontweight="bold", pad=6)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.legend(loc="lower right", frameon=False, fontsize=9)


def main():
    # Load all curves
    base_fa = _read_history_dir(RESULTS_ROOT / "flower_C0_baseline", "fedavg",  "0.0")
    base_fp = _read_history_dir(RESULTS_ROOT / "flower_C0_baseline", "fedprox", "0.01")
    base_fn = _read_history_dir(RESULTS_ROOT / "flower_C0_baseline", "fednova", "0.0")
    c1_fa   = _read_history_dir(RESULTS_ROOT / "system_het_fixed", "fedavg",  "0.0")
    c1_fp   = _read_history_dir(RESULTS_ROOT / "system_het_fixed", "fedprox", "0.01")
    c2_fa   = _read_history_dir(RESULTS_ROOT / "system_het_random", "fedavg",  "0.0")
    c2_fp   = _read_history_dir(RESULTS_ROOT / "system_het_random", "fedprox", "0.01")
    c2_fn   = _read_history_dir(RESULTS_ROOT / "system_het_random_fednova",
                                "fednova", "0.0")
    print(f"C0: n_fa={len(base_fa)}, n_fp={len(base_fp)}, n_fn={len(base_fn)}")
    print(f"C1: n_fa={len(c1_fa)}, n_fp={len(c1_fp)}")
    print(f"C2: n_fa={len(c2_fa)}, n_fp={len(c2_fp)}, n_fn={len(c2_fn)}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                             gridspec_kw={"hspace": 0.32, "wspace": 0.22})

    _plot_condition(axes[0, 0], base_fa, base_fp,
                    "(a) C0 -- no system heterogeneity (Flower baseline)")
    _plot_condition(axes[0, 1], c1_fa, c1_fp,
                    "(b) C1 -- fixed stragglers (C5, C6 always slow)")
    _plot_condition(axes[1, 0], c2_fa, c2_fp,
                    "(c) C2 -- random stragglers (4/7 clients per round, PRIMARY)")
    _plot_condition(axes[1, 1], {}, {}, fn_curves=None,
                    title="(d) FedNova catastrophic failure under C2")
    # Manually paint panel (d) with FedNova C0 vs C2
    ax = axes[1, 1]
    ax.clear()
    if base_fn:
        rounds, mean, sem = _stack(base_fn)
        ax.plot(rounds, mean, color=COLOURS["fednova"], linewidth=1.6,
                linestyle="-",  label="FedNova (C0 baseline)")
        ax.fill_between(rounds, mean - sem, mean + sem,
                        color=COLOURS["fednova"], alpha=0.15, linewidth=0)
    if c2_fn:
        rounds, mean, sem = _stack(c2_fn)
        ax.plot(rounds, mean, color=COLOURS["fednova"], linewidth=1.6,
                linestyle="--", label="FedNova (C2 random stragglers)")
        ax.fill_between(rounds, mean - sem, mean + sem,
                        color=COLOURS["fednova"], alpha=0.10, linewidth=0)
    ax.set_xlim(1, NUM_ROUNDS); ax.set_ylim(0.0, 0.62)
    ax.set_xlabel("Communication round"); ax.set_ylabel("Validation macro-F1")
    ax.set_title("(d) FedNova catastrophic failure under random stragglers",
                 loc="left", fontweight="bold", pad=6)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    fig.suptitle("Validation macro-F1 trajectories under three system-heterogeneity "
                 "conditions (Flower runtime, $n = 10$ paired seeds, mean $\\pm$ SEM)",
                 fontsize=12, y=0.995, fontweight="bold")

    for ext in ("pdf", "png"):
        out = OUT_FIG / f"F7_system_het_convergence.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"Wrote: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
