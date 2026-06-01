"""Per-class validation F1 trajectories across L4 protocols.

Connects two earlier findings:
   - Finding 11: round-20 features predict final collapse (AUC=0.80)
   - Confusion-matrix finding: rare classes get misrouted to mel-nevi

Question: WHEN during training does the collapse-vs-preservation
divergence happen? Is it early (round 10-20, validating the
early-warning detector) or late?

Specifically:
   Q1. At what round does rare-class F1 first drop below 0.05 and stay
       there? ("collapse time")
   Q2. Does FedNova maintain rare-class signal throughout, or recover
       from an initial dip?
   Q3. Is round 20 already enough to distinguish "will collapse" vs
       "will preserve" trajectories?

Reads history_*.csv files (which have val_f1_class_X per round).

Output:
   F_l4_class_trajectories_d1.{pdf,png}    Per-class F1 vs round, D1 5:1
   F_l4_class_trajectories_li2020.{pdf,png} Per-class F1, Li 2020 §5.2
   F_l4_collapse_time_distribution.{pdf,png} When does collapse happen?
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

CLASS_NAMES = ["actinic", "basal", "b-kerat", "dermato", "melanoma", "mel-nevi", "vascular"]
RARE_IDX = (3, 4, 6)
RARE_LABELS = [CLASS_NAMES[i] for i in RARE_IDX]
SEEDS = [42, 123, 456]
COLLAPSE_THRESHOLD = 0.05  # rare-class F1 below this = collapsed
EARLY_ROUND = 20  # for connecting to Finding 11


def _load_history(csv_path: Path) -> pd.DataFrame | None:
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def _mean_rare_f1_per_round(df: pd.DataFrame) -> pd.Series:
    """Mean val_f1 across rare classes per round."""
    cols = [f"val_f1_class_{c}" for c in RARE_IDX]
    if any(c not in df.columns for c in cols):
        return pd.Series(dtype=float)
    return df[cols].mean(axis=1)


def _collapse_round(df: pd.DataFrame, threshold: float = COLLAPSE_THRESHOLD) -> int | None:
    """First round where mean rare-class F1 drops below threshold AND
    stays below for the rest of training."""
    rare_mean = _mean_rare_f1_per_round(df)
    if len(rare_mean) == 0:
        return None
    rounds = df["round"].values
    for i in range(len(rare_mean)):
        if rare_mean.iloc[i] < threshold:
            # Check if it stays below for the rest
            if all(rare_mean.iloc[i:] < threshold):
                return int(rounds[i])
    return None  # never collapsed


def _avg_class_trajectory_across_seeds(
    paths: list[Path], num_rounds_cap: int = 150
) -> dict | None:
    """Average per-class F1 trajectories across seeds."""
    if not paths:
        return None
    trajs = []
    for p in paths:
        df = _load_history(p)
        if df is None or len(df) == 0:
            continue
        if "round" not in df.columns:
            continue
        df = df[df["round"] <= num_rounds_cap].copy()
        trajs.append(df)
    if not trajs:
        return None
    # Use the shortest length
    min_len = min(len(t) for t in trajs)
    rounds = trajs[0]["round"].values[:min_len]
    per_class_means = {}
    per_class_sds = {}
    for ci in range(7):
        col = f"val_f1_class_{ci}"
        stacked = np.stack([t[col].values[:min_len] for t in trajs if col in t.columns])
        per_class_means[ci] = stacked.mean(axis=0)
        per_class_sds[ci] = stacked.std(axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros(min_len)
    return dict(rounds=rounds, mean=per_class_means, sd=per_class_sds, n_seeds=len(trajs))


# ----------------------------------------------------------------
# Figure 1: D1 trajectories at 5:1 LR asymmetry
# ----------------------------------------------------------------
print("=" * 80)
print("Figure 1: D1 5:1 LR asymmetry per-class trajectories")
print("=" * 80)

D1_DIR = RESULTS / "asymmetric_lr_L4"

# Conditions: 3 algos at symmetric vs 5:1
CONDITIONS = [
    ("FedAvg 1:1", "fedavg", "0.0", ""),
    ("FedAvg 5:1", "fedavg", "0.0", "_lrPC-c0lr0.01-c1lr0.002"),
    ("FedProx 1:1", "fedprox", "0.01", ""),
    ("FedProx 5:1", "fedprox", "0.01", "_lrPC-c0lr0.01-c1lr0.002"),
    ("FedNova 1:1", "fednova", "0.0", ""),
    ("FedNova 5:1", "fednova", "0.0", "_lrPC-c0lr0.01-c1lr0.002"),
]

# 3 rows × 2 cols (algos × ratios)
fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True, sharey=True)
CLASS_COLORS_MAP = {
    0: "#88B299", 1: "#A8C09F", 2: "#C9D6C0",       # common
    3: "#E07B39",                                    # dermato (rare)
    4: "#C03A2B",                                    # melanoma (rare)
    5: "#7E9CD0",                                    # nevi (dominant)
    6: "#5E3A85",                                    # vascular (rare)
}
ALGO_ROW = {"fedavg": 0, "fedprox": 1, "fednova": 2}
RATIO_COL = {"": 0, "_lrPC-c0lr0.01-c1lr0.002": 1}

for label, algo, mu_str, tag in CONDITIONS:
    paths = [D1_DIR / f"history_{algo}_mu{mu_str}_E20{tag}_s{s}.csv" for s in SEEDS]
    res = _avg_class_trajectory_across_seeds(paths)
    if res is None:
        continue
    ax = axes[ALGO_ROW[algo], RATIO_COL[tag]]
    for ci in range(7):
        ls = "-" if ci in RARE_IDX else ":"
        lw = 1.8 if ci in RARE_IDX else 0.8
        alpha = 0.95 if ci in RARE_IDX else 0.5
        ax.plot(res["rounds"], res["mean"][ci],
                color=CLASS_COLORS_MAP[ci], linewidth=lw, linestyle=ls, alpha=alpha,
                label=CLASS_NAMES[ci] + (" (rare)" if ci in RARE_IDX else ""))
    # Mark the early-warning round
    ax.axvline(EARLY_ROUND, color="#999", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.text(EARLY_ROUND + 1, 0.95, "round 20", fontsize=7.5, color="#999", va="top")
    ax.set_title(label, loc="left", fontweight="bold", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

axes[2, 0].set_xlabel("Communication round")
axes[2, 1].set_xlabel("Communication round")
for ar in range(3):
    axes[ar, 0].set_ylabel("Validation F1")
axes[0, 1].legend(loc="upper right", frameon=False, fontsize=8, ncol=1)
fig.suptitle("D1: per-class validation F1 trajectories — rare classes (solid) vs common (dotted)",
             fontsize=11.5, fontweight="bold", y=1.005)
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"F_l4_class_trajectories_d1.{ext}")
plt.close(fig)
print(f"Wrote {OUT_FIG/'F_l4_class_trajectories_d1.pdf'}")

# ----------------------------------------------------------------
# Figure 2: Li 2020 §5.2 trajectories
# ----------------------------------------------------------------
print()
print("=" * 80)
print("Figure 2: Li 2020 §5.2 per-class trajectories")
print("=" * 80)

LI_DIR = RESULTS / "li2020_asymmetric_L4"
LI_CONDS = [
    ("FedAvg baseline (no straggler)", "fedavg",  "0.0",  ""),
    ("FedAvg + drop + straggler", "fedavg",  "0.0",  "_sh-fixed_stragglers_drop"),
    ("FedProx + γ-inexact + straggler ⭐", "fedprox", "0.01", "_sh-fixed_stragglers"),
    ("FedProx + drop (control)", "fedprox", "0.01", "_sh-fixed_stragglers_drop"),
]

fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
for col, (label, algo, mu_str, tag) in enumerate(LI_CONDS):
    paths = [LI_DIR / f"history_{algo}_mu{mu_str}_E20{tag}_s{s}.csv" for s in SEEDS]
    res = _avg_class_trajectory_across_seeds(paths)
    if res is None:
        continue
    ax = axes[col]
    for ci in range(7):
        ls = "-" if ci in RARE_IDX else ":"
        lw = 1.8 if ci in RARE_IDX else 0.8
        alpha = 0.95 if ci in RARE_IDX else 0.5
        ax.plot(res["rounds"], res["mean"][ci],
                color=CLASS_COLORS_MAP[ci], linewidth=lw, linestyle=ls, alpha=alpha,
                label=CLASS_NAMES[ci] + (" (rare)" if ci in RARE_IDX else ""))
    ax.axvline(EARLY_ROUND, color="#999", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_title(label, loc="left", fontweight="bold", fontsize=10)
    ax.set_xlabel("Round")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
axes[0].set_ylabel("Validation F1")
axes[3].legend(loc="lower right", frameon=False, fontsize=8)
fig.suptitle("Li 2020 §5.2 protocol: per-class trajectories show when γ-inexact rescues rare classes",
             fontsize=11.5, fontweight="bold", y=1.03)
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"F_l4_class_trajectories_li2020.{ext}")
plt.close(fig)
print(f"Wrote {OUT_FIG/'F_l4_class_trajectories_li2020.pdf'}")

# ----------------------------------------------------------------
# Figure 3: Collapse-time analysis across many protocols
# ----------------------------------------------------------------
print()
print("=" * 80)
print("Figure 3: Collapse-time distribution across protocols")
print("=" * 80)

# Scan all L4 history files
SOURCES = [
    ("Heterogeneity ladder",            RESULTS / "heterogeneity_ladder/L4_two_client_90_10_rare_stress", "history_fedavg_mu0.0_E20_s42.csv"),
    ("Heterogeneity ladder (FP)",       RESULTS / "heterogeneity_ladder/L4_two_client_90_10_rare_stress", "history_fedprox_mu0.01_E20_s42.csv"),
    ("Heterogeneity ladder (FN)",       RESULTS / "heterogeneity_ladder/L4_two_client_90_10_rare_stress", "history_fednova_mu0.0_E20_s42.csv"),
    ("Node-pinned (FA)",                RESULTS / "node_pinned_L4",                                       "history_fedavg_mu0.0_E20_s{s}.csv"),
    ("Node-pinned (FP)",                RESULTS / "node_pinned_L4",                                       "history_fedprox_mu0.01_E20_s{s}.csv"),
    ("Engineered 90/10 (FA)",           RESULTS / "two_client_90_10_rare_stress",                         "history_fedavg_mu0.0_E20_s{s}.csv"),
    ("Engineered 90/10 (FP)",           RESULTS / "two_client_90_10_rare_stress",                         "history_fedprox_mu0.01_E20_s{s}.csv"),
    ("Li §5.2 FA+drop",                 RESULTS / "li2020_asymmetric_L4",                                 "history_fedavg_mu0.0_E20_sh-fixed_stragglers_drop_s{s}.csv"),
    ("Li §5.2 FP+γ-inexact",            RESULTS / "li2020_asymmetric_L4",                                 "history_fedprox_mu0.01_E20_sh-fixed_stragglers_s{s}.csv"),
    ("Perfect-storm FA+drop",           RESULTS / "fedprox_perfect_storm_L4",                             "history_fedavg_mu0.0_E20_sh-random_stragglers_drop_s{s}.csv"),
    ("Perfect-storm FP μ=1.0",          RESULTS / "fedprox_perfect_storm_L4",                             "history_fedprox_mu1.0_E20_sh-random_stragglers_s{s}.csv"),
    ("Perfect-storm FP μ=0.01",         RESULTS / "fedprox_perfect_storm_L4",                             "history_fedprox_mu0.01_E20_sh-random_stragglers_s{s}.csv"),
    ("D1 1:1 FedAvg",                   D1_DIR, "history_fedavg_mu0.0_E20_s{s}.csv"),
    ("D1 5:1 FedAvg",                   D1_DIR, "history_fedavg_mu0.0_E20_lrPC-c0lr0.01-c1lr0.002_s{s}.csv"),
    ("D1 1:1 FedProx",                  D1_DIR, "history_fedprox_mu0.01_E20_s{s}.csv"),
    ("D1 5:1 FedProx",                  D1_DIR, "history_fedprox_mu0.01_E20_lrPC-c0lr0.01-c1lr0.002_s{s}.csv"),
    ("D1 1:1 FedNova",                  D1_DIR, "history_fednova_mu0.0_E20_s{s}.csv"),
    ("D1 5:1 FedNova",                  D1_DIR, "history_fednova_mu0.0_E20_lrPC-c0lr0.01-c1lr0.002_s{s}.csv"),
]

collapse_rows = []
for label, root, pattern in SOURCES:
    if "{s}" in pattern:
        seeds_to_try = SEEDS
    else:
        seeds_to_try = [None]
    times = []
    rare_at_round20s = []
    rare_finals = []
    for s in seeds_to_try:
        if s is None:
            f = root / pattern
        else:
            f = root / pattern.format(s=s)
        df = _load_history(f)
        if df is None or len(df) == 0:
            continue
        rare_mean = _mean_rare_f1_per_round(df)
        if len(rare_mean) == 0:
            continue
        ct = _collapse_round(df, COLLAPSE_THRESHOLD)
        times.append(ct if ct is not None else df["round"].max() + 1)
        # Rare F1 at round 20
        early_idx = (df["round"] - EARLY_ROUND).abs().idxmin()
        rare_at_round20s.append(float(rare_mean.iloc[early_idx]))
        rare_finals.append(float(rare_mean.iloc[-1]))
    if not times:
        continue
    collapse_rows.append(dict(
        protocol=label,
        n_seeds=len(times),
        n_collapsed=sum(1 for t in times if t <= 150),
        mean_collapse_round=float(np.mean([t for t in times if t <= 150])) if any(t <= 150 for t in times) else np.nan,
        mean_rare_r20=float(np.mean(rare_at_round20s)),
        mean_rare_final=float(np.mean(rare_finals)),
    ))
collapse_df = pd.DataFrame(collapse_rows)
collapse_df.to_csv(OUT_DIR / "l4_collapse_time_summary.csv", index=False)
print(collapse_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# Now scatter: rare F1 at round 20 vs final rare F1
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"wspace": 0.25})
# Panel A: scatter
axA = axes[0]
for _, row in collapse_df.iterrows():
    color = "#C03A2B" if row["mean_rare_final"] < 0.05 else "#1f6f3f"
    axA.scatter(row["mean_rare_r20"], row["mean_rare_final"],
                color=color, s=80, edgecolor="white", linewidth=0.8, alpha=0.85)
    axA.annotate(row["protocol"], (row["mean_rare_r20"], row["mean_rare_final"]),
                 xytext=(5, 3), textcoords="offset points", fontsize=7.5)
axA.axhline(COLLAPSE_THRESHOLD, color="#666", linestyle="--", linewidth=0.8, alpha=0.6,
            label=f"collapse threshold ({COLLAPSE_THRESHOLD})")
axA.plot([0, 0.5], [0, 0.5], color="#888", linestyle=":", linewidth=0.8, alpha=0.5, label="y=x")
axA.set_xlabel("Rare-class F1 at round 20 (early-warning feature)")
axA.set_ylabel("Final rare-class F1")
axA.set_xlim(0, 0.5)
axA.set_ylim(0, 0.6)
axA.set_title("(a) Round-20 rare F1 vs final rare F1 (validates Finding 11)",
              loc="left", fontweight="bold", fontsize=11)
axA.legend(loc="upper left", frameon=False, fontsize=9)
axA.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)

# Panel B: collapse time histogram for protocols that did collapse
axB = axes[1]
collapsed = collapse_df[collapse_df["mean_rare_final"] < 0.10].copy()
not_collapsed = collapse_df[collapse_df["mean_rare_final"] >= 0.10].copy()
if len(collapsed):
    collapsed_sorted = collapsed.sort_values("mean_collapse_round")
    y = np.arange(len(collapsed_sorted))
    axB.barh(y, collapsed_sorted["mean_collapse_round"].values,
             color="#C03A2B", edgecolor="white", linewidth=0.5, alpha=0.85)
    for i, row in enumerate(collapsed_sorted.itertuples()):
        cr = row.mean_collapse_round
        axB.text(cr + 2, i, f"round {cr:.0f}", va="center", fontsize=8)
    axB.set_yticks(y)
    axB.set_yticklabels(collapsed_sorted["protocol"], fontsize=8.5)
    axB.set_xlabel("Mean round at which rare-class F1 drops below 0.05 and stays")
    axB.set_title("(b) When does rare-class collapse happen?  —  protocols that collapsed",
                  loc="left", fontweight="bold", fontsize=11)
    axB.set_xlim(0, 160)
    axB.axvline(EARLY_ROUND, color="#888", linestyle="--", linewidth=0.8, alpha=0.6)
    axB.text(EARLY_ROUND + 2, len(collapsed_sorted) - 0.5, "round 20", fontsize=8, color="#666")
    axB.grid(True, axis="x", alpha=0.25, linestyle="--", linewidth=0.5)
    axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)

fig.suptitle("Rare-class collapse timing — when does it happen during training?",
             fontsize=11.5, fontweight="bold", y=1.02)
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"F_l4_collapse_time_distribution.{ext}")
plt.close(fig)
print(f"Wrote {OUT_FIG/'F_l4_collapse_time_distribution.pdf'}")

# ----------------------------------------------------------------
# Headline summary
# ----------------------------------------------------------------
print()
print("=" * 80)
print("HEADLINE INSIGHTS")
print("=" * 80)
print(f"  COLLAPSE_THRESHOLD = {COLLAPSE_THRESHOLD} (rare-class mean F1)")
print()
# Compute correlation between round-20 rare F1 and final rare F1
valid = collapse_df.dropna(subset=["mean_rare_r20", "mean_rare_final"])
from scipy.stats import spearmanr
sp_r, sp_p = spearmanr(valid["mean_rare_r20"], valid["mean_rare_final"])
print(f"  Spearman r (round-20 rare F1 vs final rare F1) = {sp_r:+.3f} (p = {sp_p:.4f})")
print(f"  → If r is high and significant, round 20 IS the right early-warning checkpoint")
print()
# How early does collapse happen?
collapsed_only = collapse_df[(collapse_df["mean_collapse_round"].notna()) &
                              (collapse_df["mean_collapse_round"] <= 150)]
if len(collapsed_only):
    early_collapse = collapsed_only[collapsed_only["mean_collapse_round"] <= 30]
    print(f"  Protocols that collapse:           {len(collapsed_only)} of {len(collapse_df)}")
    print(f"  Of those, collapse by round 30:    {len(early_collapse)} ({len(early_collapse)/len(collapsed_only)*100:.0f}%)")
    print(f"  Median collapse round:             {float(collapsed_only['mean_collapse_round'].median()):.0f}")
print()
print("Done.")
