"""
Generate F_val_curves_extreme_gaps.pdf — validation macro-F1 trajectories
for the two experiments showing the largest FedAvg vs FedProx differences:

  Panel A: Li 2020 §5.2 four-condition decomposition (Δ up to +0.115)
  Panel B: Perfect-storm L4 90% stragglers (Δ up to +0.404)

Each curve is the mean over 3 seeds (42, 123, 456); shaded band is ±1 SD.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def load_curve_mean_sd(file_paths, value_col="val_macro_f1", round_col="round"):
    """Load one curve per seed file, align on round, return (rounds, mean, sd)."""
    series = []
    for p in file_paths:
        df = pd.read_csv(p)
        # Aggregate to one value per round (in case of duplicates)
        agg = df.groupby(round_col)[value_col].last().sort_index()
        series.append(agg)
    common_rounds = series[0].index
    for s in series[1:]:
        common_rounds = common_rounds.intersection(s.index)
    arr = np.stack([s.loc[common_rounds].values for s in series])
    return common_rounds.values, arr.mean(axis=0), arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(arr.shape[1])


# ----------------------------------------------------------------------
# Plot setup
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.0),
                                gridspec_kw=dict(wspace=0.22))

# ----------------------------------------------------------------------
# Panel A — Li 2020 §5.2 four-condition decomposition
# ----------------------------------------------------------------------
LI = RESULTS / "li2020_asymmetric_L4"
SEEDS = ("42", "123", "456")

li_specs = [
    ("FedAvg (no straggler)",                  "#1f4f7a",     "-",
        [LI / f"history_fedavg_mu0.0_E20_s{s}.csv" for s in SEEDS]),
    ("FedAvg + drop-stragglers",               "#9d2933",     "-",
        [LI / f"history_fedavg_mu0.0_E20_sh-fixed_stragglers_drop_s{s}.csv" for s in SEEDS]),
    ("FedProx (μ=0.01) + drop (control)",      "#c97644",     "--",
        [LI / f"history_fedprox_mu0.01_E20_sh-fixed_stragglers_drop_s{s}.csv" for s in SEEDS]),
    ("FedProx (μ=0.01) + γ-inexact",           "#1e7d3a",     "-",
        [LI / f"history_fedprox_mu0.01_E20_sh-fixed_stragglers_s{s}.csv" for s in SEEDS]),
]

for label, color, ls, paths in li_specs:
    rounds, mean, sd = load_curve_mean_sd(paths)
    axA.plot(rounds, mean, color=color, linestyle=ls, linewidth=1.9, label=label)
    axA.fill_between(rounds, mean - sd, mean + sd, color=color, alpha=0.13, linewidth=0)

axA.set_xlabel("Communication round")
axA.set_ylabel("Validation macro-F1 (mean over 3 seeds; band $= \\pm 1$ SD)")
axA.set_xlim(0, 150)
axA.set_ylim(0.0, 0.62)
axA.set_xticks([0, 25, 50, 75, 100, 125, 150])
axA.grid(linestyle=":", alpha=0.4)
axA.legend(loc="lower right", framealpha=0.95, fontsize=9.5)
axA.set_title("(A) Four-condition decomposition at L4  ($\\Delta$ up to $+0.115$)",
              loc="left", fontweight="bold", fontsize=11, pad=8)

# ----------------------------------------------------------------------
# Panel B — Perfect-storm L4 (90% random stragglers)
# ----------------------------------------------------------------------
PS = RESULTS / "fedprox_perfect_storm_L4"

ps_specs = [
    ("FedAvg + drop (collapses)",                   "#9d2933",  "-",
        [PS / f"history_fedavg_mu0.0_E20_sh-random_stragglers_drop_s{s}.csv" for s in SEEDS]),
    ("FedProx (μ=1.0) + γ-inexact",                  "#c97644",  "-",
        [PS / f"history_fedprox_mu1.0_E20_sh-random_stragglers_s{s}.csv" for s in SEEDS]),
    ("FedProx (μ=0.01) + γ-inexact",                 "#1e7d3a",  "-",
        [PS / f"history_fedprox_mu0.01_E20_sh-random_stragglers_s{s}.csv" for s in SEEDS]),
]

for label, color, ls, paths in ps_specs:
    rounds, mean, sd = load_curve_mean_sd(paths)
    axB.plot(rounds, mean, color=color, linestyle=ls, linewidth=1.9, label=label)
    axB.fill_between(rounds, mean - sd, mean + sd, color=color, alpha=0.13, linewidth=0)

axB.set_xlabel("Communication round")
axB.set_ylabel("Validation macro-F1 (mean over 3 seeds; band $= \\pm 1$ SD)")
axB.set_xlim(0, 150)
axB.set_ylim(0.0, 0.62)
axB.set_xticks([0, 25, 50, 75, 100, 125, 150])
axB.grid(linestyle=":", alpha=0.4)
axB.legend(loc="lower right", framealpha=0.95, fontsize=9.5)
axB.set_title("(B) Perfect-storm L4, 90% random stragglers  ($\\Delta$ up to $+0.404$)",
              loc="left", fontweight="bold", fontsize=11, pad=8)

# Annotate final-round gap on panel B
final_round = 150
gap_text_y_offsets = {"FedAvg + drop (collapses)": 0.05,
                      "FedProx (μ=1.0) + γ-inexact": 0.36,
                      "FedProx (μ=0.01) + γ-inexact": 0.51}

fig.tight_layout(rect=[0, 0.02, 1, 0.97])

for ext in ("pdf",):
    out = OUT / f"F_val_curves_extreme_gaps.{ext}"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"wrote {out}")

plt.close(fig)
