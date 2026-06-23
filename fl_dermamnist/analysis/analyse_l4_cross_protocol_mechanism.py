"""Cross-protocol mechanism analysis on L4.

Unified analysis across ALL L4 experiments comparing per-client update
norm patterns. Builds a master table of (experiment, condition, seed,
client_id, mean_update_norm, final_macro_f1, rare_avg_f1) and asks
ONE coherent question:

   How does each protocol change the per-client INFLUENCE pattern
   (||Δw_1|| / ||Δw_0||), and does the influence pattern predict
   the rare-class outcome?

Mechanism narrative we're testing:
   "Protocols that preserve Client 1's relative influence preserve
    the rare-class signal. Protocols that suppress Client 1 (via
    drop-stragglers, asymmetric LR scaling, or proximal anchoring)
    lose rare-class F1."

L4 protocols included (all use partition two_client_90_10_rare_stress):
   - baseline (symmetric, E=20, μ=0/0.01, lr=0.01, mom=0/0.9, bs=10/32)
   - node-pinned variance isolation
   - Li 2020 §5.2 asymmetric (drop-stragglers protocol)
   - perfect-storm (Li 2020 canonical recipe at peak heterogeneity)
   - asymmetric per-client μ (Yao 2024 ablation)
   - asymmetric per-client LR (D1)
   - FedProx × weighted-CE (B2)

Outputs:
   - l4_cross_protocol_mechanism.csv : master table
   - F_l4_cross_protocol_norms.{pdf,png} : per-(experiment, algo)
        mean update norms for Client 0 vs Client 1
   - F_l4_influence_vs_outcome.{pdf,png} : scatter plot showing
        Client 1 influence ratio vs final rare-class F1
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
from fl_dermamnist.common.paths import repo_root  # noqa: E402
REPO_ROOT = repo_root()
RESULTS = REPO_ROOT / "fl_dermamnist/results"
OUT_DIR = REPO_ROOT / "fl_dermamnist/results/thesis_ready/data"
OUT_FIG = REPO_ROOT / "fl_dermamnist/results/thesis_ready/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

RARE_IDX = (3, 4, 6)


# Directory → (display label, color)
L4_EXPERIMENTS = [
    ("baseline_node_pinned",         RESULTS / "node_pinned_L4",                                    "Node-pinned symmetric",  "#7FBF94"),
    ("baseline_2c9010",              RESULTS / "two_client_90_10_rare_stress",                      "Engineered 90/10 baseline", "#88C0D0"),
    ("baseline_ladder_L4",           RESULTS / "heterogeneity_ladder/L4_two_client_90_10_rare_stress","Heterogeneity-ladder L4","#5E81AC"),
    ("li2020_asymmetric",            RESULTS / "li2020_asymmetric_L4",                              "Li 2020 §5.2 (drop+straggler)", "#BF616A"),
    ("perfect_storm",                RESULTS / "fedprox_perfect_storm_L4",                          "Perfect-storm (Li recipe)", "#D08770"),
    ("asymmetric_mu",                RESULTS / "asymmetric_mu_L4",                                   "Asymmetric per-client μ", "#EBCB8B"),
    ("asymmetric_lr",                RESULTS / "asymmetric_lr_L4",                                   "Asymmetric per-client LR (D1)", "#B48EAD"),
    ("weighted_ce",                  RESULTS / "fedprox_weighted_ce_L4",                             "FedProx × loss type (B2)", "#A3BE8C"),
]


STEM_RE = re.compile(
    r"client_update_norms_(?P<algo>fedavg|fedprox|fednova)_mu(?P<mu>[0-9.]+)_E\d+"
    r"(?P<tail>.*)_s(?P<seed>\d+)\.csv"
)


def _classify_condition(algo: str, mu: float, tail: str) -> dict:
    """Extract condition descriptors from filename tail."""
    has_drop = "_drop" in tail
    has_sh_fixed = "sh-fixed_stragglers" in tail
    has_sh_random = "sh-random_stragglers" in tail
    # Per-client μ
    pc_mu = None
    m = re.search(r"muPC-c0m([0-9.]+)-c1m([0-9.]+)", tail)
    if m:
        pc_mu = (float(m.group(1)), float(m.group(2)))
    # Per-client lr
    pc_lr = None
    m = re.search(r"lrPC-c0lr([0-9.]+)-c1lr([0-9.]+)", tail)
    if m:
        pc_lr = (float(m.group(1)), float(m.group(2)))
    # Loss type
    loss_type = "ce"
    if "loss-class_weighted_ce" in tail:
        loss_type = "weighted_ce"
    elif "loss-focal" in tail:
        loss_type = "focal"
    return dict(
        has_drop=has_drop,
        has_sh_fixed=has_sh_fixed,
        has_sh_random=has_sh_random,
        pc_mu_c0=pc_mu[0] if pc_mu else None,
        pc_mu_c1=pc_mu[1] if pc_mu else None,
        pc_lr_c0=pc_lr[0] if pc_lr else None,
        pc_lr_c1=pc_lr[1] if pc_lr else None,
        loss_type=loss_type,
    )


def _condition_label(algo: str, mu: float, cond: dict) -> str:
    """Pretty-print the condition."""
    bits = [algo]
    if mu > 0:
        bits.append(f"μ={mu}")
    if cond["pc_mu_c0"] is not None:
        bits.append(f"μ_PC=({cond['pc_mu_c0']:g},{cond['pc_mu_c1']:g})")
    if cond["pc_lr_c0"] is not None:
        # Compute ratio
        if cond["pc_lr_c1"] > 0:
            ratio = cond["pc_lr_c0"] / cond["pc_lr_c1"]
            bits.append(f"lr={ratio:.0f}:1")
    if cond["has_drop"]:
        bits.append("+drop")
    if cond["has_sh_fixed"]:
        bits.append("+sh-fixed")
    if cond["has_sh_random"]:
        bits.append("+sh-rand")
    if cond["loss_type"] != "ce":
        bits.append(f"loss={cond['loss_type']}")
    return " ".join(bits)


def _load_json_outcome(csv_path: Path) -> dict | None:
    """Find the matching test_at_best JSON to get macro-F1 + per-class."""
    json_path = csv_path.parent / csv_path.name.replace("client_update_norms_", "test_at_best_").replace(".csv", ".json")
    if not json_path.exists():
        return None
    try:
        d = json.load(open(json_path))
        pc = d.get("per_class_f1") or [float("nan")] * 7
        return dict(
            macro_f1=d.get("macro_f1"),
            rare_avg_f1=float(np.mean([pc[i] for i in RARE_IDX])),
        )
    except Exception:
        return None


# ----------------------------------------------------------------
# 1. Build the master table
# ----------------------------------------------------------------
rows = []
for exp_key, exp_dir, exp_label, exp_color in L4_EXPERIMENTS:
    if not exp_dir.exists():
        continue
    for f in exp_dir.glob("client_update_norms_*.csv"):
        m = STEM_RE.match(f.name)
        if m is None:
            continue
        algo = m.group("algo")
        mu = float(m.group("mu"))
        seed = int(m.group("seed"))
        tail = m.group("tail")
        cond = _classify_condition(algo, mu, tail)
        # Read update norm data
        df = pd.read_csv(f)
        if "client_id" not in df.columns or "update_norm" not in df.columns:
            continue
        grouped = df.groupby("client_id")["update_norm"].mean()
        if 0 not in grouped.index or 1 not in grouped.index:
            continue
        norm_c0 = float(grouped[0])
        norm_c1 = float(grouped[1])
        # Look up outcome
        outcome = _load_json_outcome(f)
        macro = outcome["macro_f1"] if outcome else np.nan
        rare = outcome["rare_avg_f1"] if outcome else np.nan
        rows.append(dict(
            experiment=exp_key,
            experiment_label=exp_label,
            experiment_color=exp_color,
            condition_label=_condition_label(algo, mu, cond),
            algorithm=algo,
            mu=mu,
            seed=seed,
            mean_norm_c0=norm_c0,
            mean_norm_c1=norm_c1,
            influence_ratio_c1_to_c0=norm_c1 / norm_c0 if norm_c0 > 0 else np.nan,
            macro_f1=macro,
            rare_avg_f1=rare,
            **cond,
        ))
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "l4_cross_protocol_mechanism.csv", index=False)
print(f"Wrote {OUT_DIR/'l4_cross_protocol_mechanism.csv'}  ({len(df)} runs across {df['experiment'].nunique()} experiments)")
print()
print("Per-experiment counts:")
print(df.groupby("experiment_label").size().to_string())

# ----------------------------------------------------------------
# 2. Per-experiment summary: mean per-client norm + influence ratio
# ----------------------------------------------------------------
summary = df.groupby(["experiment_label", "condition_label", "algorithm"]).agg(
    n=("seed", "count"),
    mean_norm_c0=("mean_norm_c0", "mean"),
    mean_norm_c1=("mean_norm_c1", "mean"),
    mean_influence=("influence_ratio_c1_to_c0", "mean"),
    sd_influence=("influence_ratio_c1_to_c0", lambda x: x.std(ddof=1) if len(x) > 1 else 0),
    mean_macro=("macro_f1", "mean"),
    mean_rare=("rare_avg_f1", "mean"),
).reset_index()
summary.to_csv(OUT_DIR / "l4_cross_protocol_summary.csv", index=False)
print()
print("=" * 110)
print("Cross-protocol summary (showing key conditions per experiment)")
print("=" * 110)
print(summary.sort_values(["experiment_label", "algorithm", "condition_label"]).to_string(index=False))

# ----------------------------------------------------------------
# 3. Headline mechanism finding: influence ratio vs rare-class outcome
# ----------------------------------------------------------------
print()
print("=" * 110)
print("HEADLINE: Does Client 1's influence ratio predict rare-class outcome?")
print("=" * 110)
# Pool all runs with valid macro_f1 and influence_ratio
valid = df.dropna(subset=["macro_f1", "rare_avg_f1", "influence_ratio_c1_to_c0"])
print(f"  {len(valid)} valid runs with both update-norm + outcome data")
from scipy.stats import spearmanr, pearsonr
sp_r, sp_p = spearmanr(valid["influence_ratio_c1_to_c0"], valid["rare_avg_f1"])
pe_r, pe_p = pearsonr(valid["influence_ratio_c1_to_c0"], valid["rare_avg_f1"])
print(f"  Influence ratio vs rare-class F1:")
print(f"    Spearman r = {sp_r:+.3f}  (p = {sp_p:.4f})")
print(f"    Pearson  r = {pe_r:+.3f}  (p = {pe_p:.4f})")
sp_r2, sp_p2 = spearmanr(valid["influence_ratio_c1_to_c0"], valid["macro_f1"])
pe_r2, pe_p2 = pearsonr(valid["influence_ratio_c1_to_c0"], valid["macro_f1"])
print(f"  Influence ratio vs macro-F1:")
print(f"    Spearman r = {sp_r2:+.3f}  (p = {sp_p2:.4f})")
print(f"    Pearson  r = {pe_r2:+.3f}  (p = {pe_p2:.4f})")

# ----------------------------------------------------------------
# 4. Figure 1: per-(experiment × algorithm) bar chart of per-client norms
# ----------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})

# Aggregate at experiment × algorithm level (collapsing conditions for the main figure)
exp_algo = df.groupby(["experiment_label", "algorithm"]).agg(
    mean_norm_c0=("mean_norm_c0", "mean"),
    sd_norm_c0=("mean_norm_c0", lambda x: x.std(ddof=1) if len(x) > 1 else 0),
    mean_norm_c1=("mean_norm_c1", "mean"),
    sd_norm_c1=("mean_norm_c1", lambda x: x.std(ddof=1) if len(x) > 1 else 0),
    n=("seed", "count"),
).reset_index().sort_values(["experiment_label", "algorithm"])

experiments_order = list(dict.fromkeys(exp_algo["experiment_label"]))
fig, ax = plt.subplots(1, 1, figsize=(15, 7))
y_positions = []
labels = []
y = 0
ALGO_COLOR = {"fedavg": "#7FBF94", "fedprox": "#3D5A80", "fednova": "#C9A227"}
bar_h = 0.35

for exp_label in experiments_order:
    sub = exp_algo[exp_algo["experiment_label"] == exp_label]
    for algo in ["fedavg", "fedprox", "fednova"]:
        row = sub[sub["algorithm"] == algo]
        if not len(row):
            continue
        c0 = float(row["mean_norm_c0"].iloc[0])
        c1 = float(row["mean_norm_c1"].iloc[0])
        c0_sd = float(row["sd_norm_c0"].iloc[0])
        c1_sd = float(row["sd_norm_c1"].iloc[0])
        n = int(row["n"].iloc[0])
        # Two side-by-side bars (C0 solid, C1 hatched)
        color = ALGO_COLOR.get(algo, "#666")
        ax.barh(y - bar_h / 2, c0, bar_h, xerr=c0_sd,
                color=color, edgecolor="white", linewidth=0.5, capsize=2, alpha=0.95)
        ax.barh(y + bar_h / 2, c1, bar_h, xerr=c1_sd,
                color=color, edgecolor="black", linewidth=0.7, capsize=2,
                alpha=0.85, hatch="//")
        labels.append(f"{exp_label}\n  {algo} (n={n})")
        y_positions.append(y)
        y += 1
    y += 0.6  # extra gap between experiments

ax.set_yticks(y_positions)
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel(r"Mean per-client update norm $\|\Delta w\|_2$  (averaged across rounds × seeds)")
ax.set_title("Per-client update-norm patterns across L4 protocols  —  "
             "solid bar = Client 0 (dominant); hatched bar = Client 1 (small specialist)",
             loc="left", fontweight="bold", fontsize=11)
ax.grid(True, axis="x", alpha=0.25, linestyle="--", linewidth=0.5)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=ALGO_COLOR["fedavg"], label="FedAvg"),
    Patch(facecolor=ALGO_COLOR["fedprox"], label="FedProx"),
    Patch(facecolor=ALGO_COLOR["fednova"], label="FedNova"),
    Patch(facecolor="#aaa", edgecolor="white", label="Client 0"),
    Patch(facecolor="#aaa", edgecolor="black", hatch="//", label="Client 1"),
]
ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=9, ncol=2)

for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"F_l4_cross_protocol_norms.{ext}")
plt.close(fig)
print()
print(f"Wrote {OUT_FIG/'F_l4_cross_protocol_norms.pdf'}")

# ----------------------------------------------------------------
# 5. Figure 2: influence ratio vs rare-class F1 scatter
# ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"wspace": 0.22})

axA, axB = axes
EXP_COLOR_MAP = {exp_label: c for _, _, exp_label, c in L4_EXPERIMENTS}

for axi, (yval_col, ytitle) in zip(axes, [
    ("rare_avg_f1", "Mean rare-class F1 (dermato, melanoma, vascular)"),
    ("macro_f1", "Test macro-F1"),
]):
    for exp_label in experiments_order:
        sub = valid[valid["experiment_label"] == exp_label]
        if not len(sub):
            continue
        color = EXP_COLOR_MAP.get(exp_label, "#666")
        axi.scatter(
            sub["influence_ratio_c1_to_c0"],
            sub[yval_col],
            color=color, s=55, edgecolor="white", linewidth=0.6,
            alpha=0.7, label=exp_label,
        )
    # Compute correlation
    r_s, p_s = spearmanr(valid["influence_ratio_c1_to_c0"], valid[yval_col])
    axi.set_xlabel(r"Client 1 / Client 0 update-norm ratio (influence)")
    axi.set_ylabel(ytitle)
    axi.set_xscale("log")
    axi.set_title(f"{ytitle} vs Client 1 influence\nSpearman r = {r_s:+.3f}  (p = {p_s:.4f})",
                  loc="left", fontweight="bold", fontsize=10)
    axi.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    axi.spines["top"].set_visible(False); axi.spines["right"].set_visible(False)

axA.legend(loc="upper left", frameon=False, fontsize=7, ncol=2)
fig.suptitle("L4 cross-protocol mechanism: Client 1's relative update-norm influence "
             "predicts rare-class outcome",
             fontsize=11.5, fontweight="bold", y=1.02)
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"F_l4_influence_vs_outcome.{ext}")
plt.close(fig)
print(f"Wrote {OUT_FIG/'F_l4_influence_vs_outcome.pdf'}")
print()
print("Done.")
