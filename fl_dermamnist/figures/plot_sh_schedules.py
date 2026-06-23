"""System-heterogeneity schedule visualisation -- clean 1x3 heatmap grid.

Reconstructs the per-(round, client) local-epoch schedule for each of
the three system-heterogeneity conditions (C0 uniform, C1 fixed
stragglers, C2 random stragglers) at seed=42. Each schedule is a
deterministic function of the seed via fl/system_het.py, so paired
FedAvg/FedProx runs at the same seed see identical schedules.

Output:
  results/thesis_ready/figures/F_sh_schedules.{pdf,png}

Style: matches F_iid_convergence, F_engineered_convergence,
F_partition_robustness, F_update_norms -- axes + panel subtitles only,
no figure title, no inset notes. LaTeX caption carries the description.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fl_dermamnist.fl.system_het import SystemHetConfig, build_epoch_schedule


THIS = Path(__file__).resolve()
from fl_dermamnist.common.paths import repo_root  # noqa: E402
REPO_ROOT = repo_root()
OUT_FIG = REPO_ROOT / "fl_dermamnist" / "results" / "thesis_ready" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

NUM_CLIENTS = 7
NUM_ROUNDS = 150
SEED = 42


plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":         11,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})


def _schedule(mode: str) -> np.ndarray:
    cfg = SystemHetConfig(
        mode=mode,
        E_max=20,
        E_straggler=5,
        fixed_straggler_ids=[5, 6] if mode == "fixed_stragglers" else None,
        random_straggler_fraction=0.5,
        random_straggler_min_epochs=1,
        random_straggler_max_epochs=None,
    )
    return build_epoch_schedule(
        cfg, num_clients=NUM_CLIENTS, num_rounds=NUM_ROUNDS, seed=SEED,
    )


def main():
    c0 = _schedule("uniform")
    c1 = _schedule("fixed_stragglers")
    c2 = _schedule("random_stragglers")

    fig, axes = plt.subplots(1, 3, figsize=(14, 6.5),
                             gridspec_kw={"wspace": 0.22})

    panels = [
        ("C0 -- no system heterogeneity",    c0),
        ("C1 -- fixed stragglers",           c1),
        ("C2 -- random stragglers",          c2),
    ]

    im = None
    for ax, (title, sched) in zip(axes, panels):
        im = ax.imshow(sched, aspect="auto", cmap="viridis",
                       vmin=0, vmax=20, interpolation="nearest",
                       origin="lower")
        ax.set_xticks(range(NUM_CLIENTS))
        ax.set_xticklabels([f"C{i}" for i in range(NUM_CLIENTS)], fontsize=10)
        ax.set_xlabel("Client")
        ax.set_title(title, loc="left", pad=4, fontsize=11, fontweight="bold")

    axes[0].set_ylabel("Communication round")

    # Shared colourbar at the bottom
    fig.subplots_adjust(left=0.06, right=0.97, top=0.93, bottom=0.18)
    cbar_ax = fig.add_axes([0.18, 0.07, 0.66, 0.025])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Local epochs assigned to (round, client)")

    for ext in ("pdf", "png"):
        out = OUT_FIG / f"F_sh_schedules.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"Wrote: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
