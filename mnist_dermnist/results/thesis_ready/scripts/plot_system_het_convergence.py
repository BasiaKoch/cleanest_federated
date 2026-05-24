"""System-heterogeneity convergence curves -- F7 (cross-partition).

2x3 layout:
  Top row:    Engineered partition, C0 / C1 / C2
  Bottom row: IID partition,        C0 / C1 / C2

Each panel shows FedAvg vs FedProx mean validation macro-F1 +/- SEM
across the 10 paired seeds.

Output:
  results/thesis_ready/figures/F7_system_het_convergence.{pdf,png}
"""
from __future__ import annotations

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
    "fedavg":  "#7FBF94",
    "fedprox": "#3D5A80",
}
LABEL = {
    "fedavg":  "FedAvg",
    "fedprox": "FedProx ($\\mu=0.01$)",
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":   11,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})


def _read_history_dir(directory: Path, algo: str, mu: str) -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    pat = re.compile(rf"history_{algo}_mu{mu}_E20"
                     rf"(?:_sh-[a-z_]+)?_s(\d+)\.csv")
    for f in sorted(directory.glob(f"history_{algo}_*.csv")):
        m = pat.match(f.name)
        if not m:
            continue
        seed = int(m.group(1))
        df = pd.read_csv(f)
        out[seed] = df[["round", "val_macro_f1"]].copy()
    return out


def _stack(curves: dict[int, pd.DataFrame]):
    if not curves:
        return np.array([]), np.array([]), np.array([])
    rounds = np.arange(1, NUM_ROUNDS + 1)
    mat = np.full((len(curves), NUM_ROUNDS), np.nan)
    for i, (_, df) in enumerate(curves.items()):
        for _, row in df.iterrows():
            r = int(row["round"])
            if 1 <= r <= NUM_ROUNDS:
                mat[i, r - 1] = float(row["val_macro_f1"])
    mean = np.nanmean(mat, axis=0)
    sem = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(mat), axis=0))
    return rounds, mean, sem


def _plot_panel(ax, fa_curves, fp_curves, title):
    for algo, curves in [("fedavg", fa_curves), ("fedprox", fp_curves)]:
        if not curves:
            continue
        rounds, mean, sem = _stack(curves)
        ax.plot(rounds, mean, color=COLOURS[algo], linewidth=1.5, label=LABEL[algo])
        ax.fill_between(rounds, mean - sem, mean + sem,
                        color=COLOURS[algo], alpha=0.18, linewidth=0)
    ax.set_xlim(1, NUM_ROUNDS)
    ax.set_ylim(0.0, 0.65)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title(title, loc="left", fontweight="bold", pad=6, fontsize=10)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.legend(loc="lower right", frameon=False, fontsize=9)


def main():
    # Engineered partition arms
    eng = {
        ("C0", "no system het, baseline"): (
            _read_history_dir(RESULTS_ROOT / "flower_C0_baseline", "fedavg",  "0.0"),
            _read_history_dir(RESULTS_ROOT / "flower_C0_baseline", "fedprox", "0.01"),
        ),
        ("C1", "fixed stragglers (C5, C6 @ E=5)"): (
            _read_history_dir(RESULTS_ROOT / "system_het_fixed", "fedavg",  "0.0"),
            _read_history_dir(RESULTS_ROOT / "system_het_fixed", "fedprox", "0.01"),
        ),
        ("C2", "random stragglers (4/7 per round)"): (
            _read_history_dir(RESULTS_ROOT / "system_het_random", "fedavg",  "0.0"),
            _read_history_dir(RESULTS_ROOT / "system_het_random", "fedprox", "0.01"),
        ),
    }
    iid = {
        ("C0", "no system het, baseline"): (
            _read_history_dir(RESULTS_ROOT / "flower_C0_iid_baseline", "fedavg",  "0.0"),
            _read_history_dir(RESULTS_ROOT / "flower_C0_iid_baseline", "fedprox", "0.01"),
        ),
        ("C1", "fixed stragglers (C5, C6 @ E=5)"): (
            _read_history_dir(RESULTS_ROOT / "system_het_iid_fixed", "fedavg",  "0.0"),
            _read_history_dir(RESULTS_ROOT / "system_het_iid_fixed", "fedprox", "0.01"),
        ),
        ("C2", "random stragglers (4/7 per round)"): (
            _read_history_dir(RESULTS_ROOT / "system_het_iid_random", "fedavg",  "0.0"),
            _read_history_dir(RESULTS_ROOT / "system_het_iid_random", "fedprox", "0.01"),
        ),
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 9),
                             gridspec_kw={"hspace": 0.34, "wspace": 0.22})

    for col, ((cond, desc), (fa, fp)) in enumerate(eng.items()):
        _plot_panel(axes[0, col], fa, fp, f"(engineered) {cond} -- {desc}")
    for col, ((cond, desc), (fa, fp)) in enumerate(iid.items()):
        _plot_panel(axes[1, col], fa, fp, f"(IID) {cond} -- {desc}")

    fig.suptitle(
        "Validation macro-F1 trajectories under three system-heterogeneity "
        "conditions on each of two partition designs\n"
        "(Flower runtime, $n = 10$ paired seeds, mean $\\pm$ SEM band)",
        fontsize=12, y=1.00, fontweight="bold"
    )

    for ext in ("pdf", "png"):
        out = OUT_FIG / f"F7_system_het_convergence.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"Wrote: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
