"""Experiment A3 — Plasticity-stability Pareto frontier for μ.

Adapts the plasticity-stability framing from federated continual
learning (DOLFIN, arXiv:2510.13567, 2025; Pareto Continual Learning,
arXiv:2503.23390, 2025) to single-task FedProx. For each μ value at
each partition, we decompose per-class F1 trajectories into:

  plasticity_c  = Σ_t max(0, F1_c[t+1] - F1_c[t])     (productive learning)
  forgetting_c  = Σ_t max(0, F1_c[t] - F1_c[t+1])     (catastrophic forgetting)

Aggregating across classes:
  plasticity   = Σ_c plasticity_c
  forgetting   = Σ_c forgetting_c
  net_gain     = plasticity - forgetting

The conjecture: μ controls a plasticity-stability tradeoff.
  μ = 0           → high plasticity, high forgetting (noisy)
  μ → ∞           → low plasticity, low forgetting (stuck)
  μ optimal       → knee of the Pareto curve

Why this is a contribution:
  - DOLFIN and Pareto Continual Learning framed this for FEDERATED
    continual learning (sequential tasks)
  - Nobody has applied the framing to SINGLE-TASK FedProx, where μ
    is the natural plasticity knob
  - This unifies two perspectives: the μ that maximises final-round
    macro-F1 should correspond to the knee of the Pareto curve

Output:
  - plasticity_stability_summary.csv : per-(partition, μ) decomposition
  - F_plasticity_stability_pareto.{pdf,png}: Pareto frontier visualisation
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[4]
RESULTS = REPO_ROOT / "mnist_dermnist/results"
OUT_DIR = REPO_ROOT / "mnist_dermnist/results/thesis_ready/data"
OUT_FIG = REPO_ROOT / "mnist_dermnist/results/thesis_ready/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

CLASS_F1_COLS = [f"val_f1_class_{c}" for c in range(7)]
RARE_IDX = (3, 4, 6)


# Source dirs: μ-sweep ladder data + heterogeneity_ladder data
# (μ-sweep is the cleanest comparison; ladder has FedAvg=μ=0 baseline)
LEVEL_DIRS = {
    "L0 (IID 50/50)":         RESULTS / "mu_sweep_ladder/L0_two_client_50_50_stratified_iid",
    "L2 (label-skew 50/50)":   RESULTS / "mu_sweep_ladder/L2_two_client_50_50_label_skew_only",
    "L4 (severe 90/10)":       RESULTS / "mu_sweep_ladder/L4_two_client_90_10_rare_stress",
}
LEVEL_FEDAVG_DIRS = {
    "L0 (IID 50/50)":         RESULTS / "heterogeneity_ladder/L0_two_client_50_50_stratified_iid",
    "L2 (label-skew 50/50)":   RESULTS / "heterogeneity_ladder/L2_two_client_50_50_label_skew_only",
    "L4 (severe 90/10)":       RESULTS / "heterogeneity_ladder/L4_two_client_90_10_rare_stress",
}

STEM_RE = re.compile(
    r"history_(?P<algo>fedavg|fedprox|fednova)_mu(?P<mu>[0-9.]+)_E\d+(?P<tail>.*)_s(?P<seed>\d+)\.csv"
)


def _parse(path: Path) -> dict | None:
    m = STEM_RE.match(path.name)
    if m is None:
        return None
    return dict(algorithm=m.group("algo"), mu=float(m.group("mu")),
                seed=int(m.group("seed")), tail=m.group("tail"))


def _decompose(df: pd.DataFrame) -> dict:
    """Compute plasticity, forgetting per class, then sum across classes."""
    if any(c not in df.columns for c in CLASS_F1_COLS) or len(df) < 2:
        return dict()
    plasticity_per_class = []
    forgetting_per_class = []
    final_per_class = []
    rare_plast = 0.0
    rare_forget = 0.0
    for ci, col in enumerate(CLASS_F1_COLS):
        s = df[col].dropna().values
        if len(s) < 2:
            plasticity_per_class.append(0.0)
            forgetting_per_class.append(0.0)
            final_per_class.append(s[-1] if len(s) else 0.0)
            continue
        d = np.diff(s)
        plast_c = float(np.sum(np.maximum(0.0, d)))
        forget_c = float(np.sum(np.maximum(0.0, -d)))
        plasticity_per_class.append(plast_c)
        forgetting_per_class.append(forget_c)
        final_per_class.append(float(s[-1]))
        if ci in RARE_IDX:
            rare_plast += plast_c
            rare_forget += forget_c

    return dict(
        plasticity=float(sum(plasticity_per_class)),
        forgetting=float(sum(forgetting_per_class)),
        net_gain=float(sum(plasticity_per_class) - sum(forgetting_per_class)),
        rare_plasticity=rare_plast,
        rare_forgetting=rare_forget,
        final_macro_f1=float(np.mean(final_per_class)),
        n_rounds=len(df),
    )


# ----------------------------------------------------------------
# 1. Build the table: one row per (level, algo, μ)
# ----------------------------------------------------------------
rows = []
for level_label, root in LEVEL_DIRS.items():
    if not root.exists():
        print(f"  skip (no dir): {level_label}")
        continue
    # FedProx runs at multiple μ values (from μ-sweep)
    for f in root.glob("history_fedprox_*.csv"):
        meta = _parse(f)
        if meta is None:
            continue
        df = pd.read_csv(f)
        d = _decompose(df)
        if d:
            rows.append(dict(level=level_label, **meta, **d, file=f.name))
    # FedAvg baseline from heterogeneity_ladder/
    fa_root = LEVEL_FEDAVG_DIRS.get(level_label)
    if fa_root and fa_root.exists():
        for f in fa_root.glob("history_fedavg_*.csv"):
            meta = _parse(f)
            if meta is None:
                continue
            df = pd.read_csv(f)
            d = _decompose(df)
            if d:
                rows.append(dict(level=level_label, **meta, **d, file=f.name))

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "plasticity_stability_summary.csv", index=False)
print(f"Wrote {OUT_DIR/'plasticity_stability_summary.csv'}  ({len(df)} runs)")
print()
print(df[["level", "algorithm", "mu", "plasticity", "forgetting", "net_gain",
         "rare_plasticity", "rare_forgetting", "final_macro_f1"]].to_string(index=False))

# ----------------------------------------------------------------
# 2. For each level, identify the Pareto-optimal μ (highest net_gain)
# ----------------------------------------------------------------
print()
print("=" * 80)
print("Pareto-optimal μ per partition (maximises plasticity − forgetting):")
print("=" * 80)
for level in df["level"].unique():
    sub = df[df["level"] == level].sort_values("net_gain", ascending=False)
    if not len(sub):
        continue
    best = sub.iloc[0]
    print(f"  {level:<28}  best μ* = {best['mu']:>5}  "
          f"net = {best['net_gain']:+.3f}  "
          f"(plasticity = {best['plasticity']:.3f}, forgetting = {best['forgetting']:.3f})")

# ----------------------------------------------------------------
# 3. Figure: 2-panel Pareto visualisation
# ----------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.5),
                                gridspec_kw={"wspace": 0.25})

# Panel A: plasticity-forgetting scatter, one cluster per partition
LEVEL_COLORS = {"L0 (IID 50/50)": "#7FBF94",
                "L2 (label-skew 50/50)": "#3D5A80",
                "L4 (severe 90/10)": "#C03A2B"}
MU_MARKERS = {0.0: "X", 0.001: "o", 0.01: "s", 0.1: "D", 1.0: "^"}

for level in df["level"].unique():
    sub = df[df["level"] == level].sort_values("mu")
    if not len(sub):
        continue
    color = LEVEL_COLORS.get(level, "#444")
    for _, row in sub.iterrows():
        marker = MU_MARKERS.get(row["mu"], "o")
        algo = row["algorithm"]
        size = 130 if algo == "fedavg" else 90
        edgecolor = "black" if algo == "fedavg" else "white"
        axA.scatter(row["plasticity"], row["forgetting"],
                    color=color, marker=marker, s=size,
                    edgecolor=edgecolor, linewidth=1.2, alpha=0.85, zorder=3)
        axA.annotate(
            f"μ={row['mu']}" if algo == "fedprox" else "FedAvg (μ=0)",
            (row["plasticity"], row["forgetting"]),
            xytext=(7, 3), textcoords="offset points",
            fontsize=7.5, color=color, alpha=0.85,
        )
    # Connect points by μ ordering
    fp = sub[sub["algorithm"] == "fedprox"].sort_values("mu")
    if len(fp) > 1:
        axA.plot(fp["plasticity"], fp["forgetting"],
                 color=color, linewidth=1.2, linestyle="--", alpha=0.5, zorder=2)
axA.set_xlabel("Plasticity Σ Σ_c max(0, ΔF1_c)  (productive per-class learning)")
axA.set_ylabel("Forgetting Σ Σ_c max(0, -ΔF1_c)  (per-class regress)")
axA.set_title("(a) Plasticity-forgetting Pareto cloud — colour = partition, marker = μ",
              loc="left", fontweight="bold", fontsize=11)
axA.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)

# Custom legend
from matplotlib.lines import Line2D
level_handles = [Line2D([0], [0], marker="s", color="w",
                         markerfacecolor=c, markeredgecolor="w",
                         markersize=10, label=lab)
                  for lab, c in LEVEL_COLORS.items()]
mu_handles = [Line2D([0], [0], marker=m, color="w", markerfacecolor="#666",
                      markeredgecolor="w", markersize=9, label=f"μ={mu}")
               for mu, m in MU_MARKERS.items()]
axA.legend(handles=level_handles + mu_handles, loc="upper left",
           frameon=False, fontsize=8, ncol=2)

# Panel B: net_gain vs μ per partition
for level in df["level"].unique():
    sub = df[(df["level"] == level) & (df["algorithm"] == "fedprox")].sort_values("mu")
    if not len(sub):
        continue
    color = LEVEL_COLORS.get(level, "#444")
    axB.plot(sub["mu"], sub["net_gain"], "-o",
             color=color, linewidth=1.5, markersize=9,
             label=level)
    # Mark optimum
    best = sub.loc[sub["net_gain"].idxmax()]
    axB.scatter(best["mu"], best["net_gain"], color=color, s=200,
                edgecolor="black", linewidth=1.5, marker="*", zorder=5)
# FedAvg as horizontal baseline per partition
for level in df["level"].unique():
    fa = df[(df["level"] == level) & (df["algorithm"] == "fedavg")]
    if len(fa):
        ng = float(fa["net_gain"].iloc[0])
        axB.axhline(ng, color=LEVEL_COLORS.get(level, "#444"),
                    linestyle=":", linewidth=1.0, alpha=0.6)

axB.set_xscale("log")
axB.set_xlabel(r"FedProx $\mu$ (log scale)")
axB.set_ylabel("Net gain  (plasticity − forgetting)")
axB.set_title("(b) Net-gain Pareto curves — star marks μ*; dotted = FedAvg baseline",
              loc="left", fontweight="bold", fontsize=11)
axB.legend(loc="lower left", frameon=False, fontsize=9)
axB.grid(True, which="both", alpha=0.25, linestyle="--", linewidth=0.5)
axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)

fig.suptitle("Plasticity-stability framing for FedProx μ — adapted from DOLFIN (arXiv:2510.13567, 2025)",
             fontsize=11.5, fontweight="bold", y=1.02)
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"F_plasticity_stability_pareto.{ext}")
plt.close(fig)
print()
print(f"Wrote {OUT_FIG/'F_plasticity_stability_pareto.pdf'}")
print()
print("Done.")
