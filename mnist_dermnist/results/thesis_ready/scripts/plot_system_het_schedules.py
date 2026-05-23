"""System-heterogeneity epoch-schedule visualisation -- F6.

Mirrors the role of F5 (statistical-heterogeneity partition distribution)
for the system-heterogeneity arm of the experimental matrix. Shows the
per-(round, client) local-epoch budget under each of the three conditions.

Layout: 1x3 row of heatmaps, one per condition.
  (a) C0 (no system het)           -- uniform E_max=20 for all (round, client)
  (b) C1 (fixed stragglers)        -- clients C5 and C6 always at E=5
  (c) C2 (random stragglers)       -- 4 of 7 clients per round at
                                       E_i ~ Uniform{1..19}; others at E_max=20

Each panel:
  Rows    = 150 communication rounds
  Columns = 7 clients (C0..C6)
  Cell    = local-epoch count assigned to client k in round r
  Colour: viridis, 0 (black) to 20 (yellow)

The schedules are reconstructed deterministically from seed=42 using
mnist_dermnist.fl.system_het.build_epoch_schedule, so this figure shows
exactly the per-round work assignment that the experiments executed.

Output:
  results/thesis_ready/figures/F6_system_het_schedules.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mnist_dermnist.fl.system_het import SystemHetConfig, build_epoch_schedule


THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[4]
OUT_FIG = REPO_ROOT / "mnist_dermnist" / "results" / "thesis_ready" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

NUM_CLIENTS = 7
NUM_ROUNDS = 150
SEED = 42


plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":   11,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
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

    fig, axes = plt.subplots(1, 3, figsize=(15, 7),
                             gridspec_kw={"wspace": 0.30})

    panels = [
        ("(a) C0 -- no system heterogeneity\n      all clients $E = 20$ every round",      c0),
        ("(b) C1 -- fixed stragglers\n      C5, C6 always $E = 5$",                         c1),
        ("(c) C2 -- random stragglers (primary)\n      4/7 clients per round, $E \\sim U\\{1..19\\}$", c2),
    ]

    for ax, (title, sched) in zip(axes, panels):
        im = ax.imshow(sched, aspect="auto", cmap="viridis",
                       vmin=0, vmax=20, interpolation="nearest",
                       origin="lower")
        ax.set_xticks(range(NUM_CLIENTS))
        ax.set_xticklabels([f"C{i}" for i in range(NUM_CLIENTS)], fontsize=9)
        ax.set_xlabel("Client")
        ax.set_ylabel("Communication round")
        ax.set_title(title, loc="left", fontweight="bold", pad=8)

        # Per-condition descriptive stats
        n_straggler_cells = int((sched < 20).sum())
        total_cells = sched.size
        mean_e = float(sched.mean())
        ax.text(0.5, -0.18,
                f"straggler cells: {n_straggler_cells}/{total_cells} "
                f"({100 * n_straggler_cells / total_cells:.0f}\\%), "
                f"mean $E = {mean_e:.2f}$",
                transform=ax.transAxes, ha="center", fontsize=8, style="italic")

    # Shared colourbar
    cbar = fig.colorbar(im, ax=axes, orientation="horizontal",
                        fraction=0.04, pad=0.18, aspect=40)
    cbar.set_label("Local epochs assigned to (round, client)")

    fig.suptitle("Per-(round, client) local-epoch schedules under the three "
                 "system-heterogeneity conditions (seed 42; deterministic given seed)",
                 fontsize=12, y=1.02, fontweight="bold")

    for ext in ("pdf", "png"):
        out = OUT_FIG / f"F6_system_het_schedules.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"Wrote: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
