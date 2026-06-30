"""Update-norm mechanism diagnostic figure -- clean two-panel curve-vs-curve.

Panel A (engineered non-IID partition):
  Reads per-client per-round update-norm CSVs from the engineered Flower sweep:
    results/flower_C0_baseline/
  Mean (across clients within each round) per-round update-norm trajectory for
  FedAvg vs FedProx, averaged across paired seeds with a light +/-1 SEM band.
  FedProx tracks lower throughout (round-mean ~1.029 vs ~1.523, -32%).

Panel B (IID mechanism-null partition):
  Same quantity on the IID partition (results/flower_C0_iid_baseline/), where the
  drift-control mechanism should be largely inactive: the two trajectories sit low
  and close together (round-mean ~0.724 vs ~0.825).

Both panels share the same y-axis scale, x-axis range, colours and line styles so
they are directly comparable. This figure is a supporting diagnostic, not causal
proof; the cross-protocol scatter / Spearman correlations live in the prose, not
here.

Output:
  results/thesis_ready/figures/F_update_norms.{pdf,png}
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
Y_MAX = 5.5  # shared across both panels (global max mean+SEM ~5.25)

# FedAvg / FedProx colours, consistent with the other convergence figures.
COL_FEDAVG  = "#7FBF94"
COL_FEDPROX = "#3D5A80"

# distinct line styles give a second, colour-blind-safe cue, identical in both panels.
STYLE = {
    "fedavg":  dict(color=COL_FEDAVG,  linestyle="-",  linewidth=1.4,
                    label="FedAvg"),
    "fedprox": dict(color=COL_FEDPROX, linestyle="--", linewidth=1.4,
                    label="FedProx ($\\mu = 0.01$)"),
}


plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":           11,
    "axes.facecolor":     "white",
    "figure.facecolor":   "white",
    "savefig.dpi":         300,
    "savefig.facecolor":  "white",
    "savefig.bbox":       "tight",
    "savefig.pad_inches":  0.05,
})


def load_per_round(sweep_dir: Path, algo: str, mu: str) -> dict[int, np.ndarray]:
    """Per-seed array of length NUM_ROUNDS: mean update norm across clients per round."""
    out: dict[int, np.ndarray] = {}
    pat = re.compile(rf"client_update_norms_{algo}_mu{mu}_E20_s(\d+)\.csv")
    for f in sorted(sweep_dir.glob(f"client_update_norms_{algo}_*.csv")):
        m = pat.match(f.name)
        if not m:
            continue
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
    """Return (rounds, mean, sem) across seeds; SEM = std / sqrt(n_seeds present)."""
    rounds = np.arange(1, NUM_ROUNDS + 1)
    if not curves:
        return rounds, np.full(NUM_ROUNDS, np.nan), np.full(NUM_ROUNDS, np.nan)
    mat = np.stack(list(curves.values()))
    mean = np.nanmean(mat, axis=0)
    sem = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(mat), axis=0))
    return rounds, mean, sem


def _traj_panel(ax, fa, fp, *, title, show_legend=False, show_ylabel=False):
    """Plot FedAvg / FedProx mean trajectories with light +/-1 SEM bands."""
    for curves, style in [(fa, STYLE["fedavg"]), (fp, STYLE["fedprox"])]:
        rounds, mean, sem = stack(curves)
        ax.fill_between(rounds, mean - sem, mean + sem,
                        color=style["color"], alpha=0.15, linewidth=0)
        ax.plot(rounds, mean, **style)

    ax.set_xlim(1, NUM_ROUNDS)
    ax.set_ylim(0, Y_MAX)
    ax.set_xlabel("Communication round")
    if show_ylabel:
        ax.set_ylabel(r"Mean client update norm $\|\Delta w\|_2$")

    # restrained styling: light horizontal guides only, clean spines.
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.set_title(title, fontsize=11, loc="left", pad=6)
    if show_legend:
        ax.legend(loc="upper right", frameon=False, fontsize=10)


def main():
    eng_fa = load_per_round(RESULTS_ROOT / "flower_C0_baseline",     "fedavg",  "0.0")
    eng_fp = load_per_round(RESULTS_ROOT / "flower_C0_baseline",     "fedprox", "0.01")
    iid_fa = load_per_round(RESULTS_ROOT / "flower_C0_iid_baseline", "fedavg",  "0.0")
    iid_fp = load_per_round(RESULTS_ROOT / "flower_C0_iid_baseline", "fedprox", "0.01")
    print(f"Engineered: n_fa={len(eng_fa)}, n_fp={len(eng_fp)}")
    print(f"IID:        n_fa={len(iid_fa)}, n_fp={len(iid_fp)}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True,
                             gridspec_kw={"wspace": 0.08})

    _traj_panel(axes[0], eng_fa, eng_fp,
                title="(A) Engineered non-IID partition",
                show_legend=True, show_ylabel=True)
    _traj_panel(axes[1], iid_fa, iid_fp,
                title="(B) IID mechanism-null partition",
                show_legend=False, show_ylabel=False)

    # One unobtrusive numeric annotation in Panel A (drift-control magnitude).
    eng_fa_mean = np.nanmean(stack(eng_fa)[1])
    eng_fp_mean = np.nanmean(stack(eng_fp)[1])
    drop = 100.0 * (eng_fa_mean - eng_fp_mean) / eng_fa_mean
    axes[0].text(
        0.97, 0.74,
        f"mean norm: {eng_fa_mean:.3f} $\\to$ {eng_fp_mean:.3f} "
        f"($-{drop:.0f}\\%$)",
        transform=axes[0].transAxes, ha="right", va="top",
        fontsize=8.5, color="0.30",
    )

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_FIG / f"F_update_norms.{ext}"
        fig.savefig(out)
        print(f"Wrote: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
