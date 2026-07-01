"""
Generate F_val_curves_l4_four_condition.pdf -- single-panel version of the
Li 2020 §5.2 four-condition validation macro-F1 trajectories. This is the
main-text figure for §3.3; the perfect-storm Panel B (previously the
second subplot of F_val_curves_extreme_gaps.pdf) is appendix-only per the
build guide and is preserved in the original two-panel PDF.

Each curve is the mean over 3 seeds (42, 123, 456); shaded band is +/- 1 SD.
"""
from pathlib import Path
from fl_dermamnist.common.paths import repo_root, package_root, results_root, thesis_ready_root, thesis_data_dir, thesis_figures_dir  # noqa: E402
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = package_root()
RESULTS = ROOT / "results"
OUT = thesis_figures_dir()
OUT.mkdir(parents=True, exist_ok=True)


def load_curve_mean_sd(file_paths, value_col="val_macro_f1", round_col="round"):
    """Load one curve per seed file, align on round, return (rounds, mean, sd)."""
    series = []
    for p in file_paths:
        df = pd.read_csv(p)
        agg = df.groupby(round_col)[value_col].last().sort_index()
        series.append(agg)
    common_rounds = series[0].index
    for s in series[1:]:
        common_rounds = common_rounds.intersection(s.index)
    arr = np.stack([s.loc[common_rounds].values for s in series])
    return (common_rounds.values,
            arr.mean(axis=0),
            arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(arr.shape[1]))


plt.rcParams.update({
    "font.family": "serif",
    "font.size":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(7.2, 4.6))

LI = RESULTS / "li2020_asymmetric_L4"
SEEDS = ("42", "123", "456")

li_specs = [
    ("FedAvg (no straggler)",                  "#1f4f7a",     "-",
        [LI / f"history_fedavg_mu0.0_E20_s{s}.csv" for s in SEEDS]),
    ("FedAvg + drop-stragglers",               "#9d2933",     "-",
        [LI / f"history_fedavg_mu0.0_E20_sh-fixed_stragglers_drop_s{s}.csv" for s in SEEDS]),
    ("FedProx ($\\mu=0.01$) + drop (control)", "#c97644",     "--",
        [LI / f"history_fedprox_mu0.01_E20_sh-fixed_stragglers_drop_s{s}.csv" for s in SEEDS]),
    ("FedProx ($\\mu=0.01$) + $\\gamma$-inexact", "#1e7d3a",  "-",
        [LI / f"history_fedprox_mu0.01_E20_sh-fixed_stragglers_s{s}.csv" for s in SEEDS]),
]

for label, color, ls, paths in li_specs:
    rounds, mean, sd = load_curve_mean_sd(paths)
    ax.plot(rounds, mean, color=color, linestyle=ls, linewidth=1.9, label=label)
    ax.fill_between(rounds, mean - sd, mean + sd, color=color, alpha=0.13, linewidth=0)

ax.set_xlabel("Communication round")
ax.set_ylabel("Validation macro-F1 (mean over 3 seeds; band $= \\pm 1$ SD)")
ax.set_xlim(0, 150)
ax.set_ylim(0.0, 0.62)
ax.set_xticks([0, 25, 50, 75, 100, 125, 150])
ax.grid(linestyle=":", alpha=0.4)
ax.legend(loc="lower right", framealpha=0.95, fontsize=9.5)
ax.set_title("Li 2020 §5.2 four-condition decomposition at L4  "
             "($\\Delta$ up to $+0.115$)",
             loc="left", fontweight="bold", fontsize=10.5, pad=8)

fig.tight_layout()
out = OUT / "F_val_curves_l4_four_condition.pdf"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"wrote {out}")
plt.close(fig)
