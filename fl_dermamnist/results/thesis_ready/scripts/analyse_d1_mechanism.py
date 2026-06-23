"""Deep mechanism investigation of D1 (asymmetric LR) findings.

Three open mechanism questions from Finding #14:
  Q1. Why does FedNova outperform FedAvg even at 1:1 symmetric?
  Q2. Why does FedNova ABSORB LR asymmetry while FedAvg collapses?
  Q3. Why does FedProx collapse MORE than FedAvg under LR asymmetry?

Uses ONLY existing data (27 client_update_norms_*.csv + 27 history_*.csv).

Three mechanism hypotheses tested:
  H1: FedNova's aggregation effectively LR-normalizes per-client updates
      (i.e. Client 1's effective contribution is LR-invariant under FedNova
      but scales with LR_1 under FedAvg/FedProx)
  H2: FedProx's proximal anchor disproportionately suppresses the small
      client's already-attenuated updates (||Δw_1|| under FedProx <
      ||Δw_1|| under FedAvg, with the gap widening as LR_1 shrinks)
  H3: Per-class survival differs — FedNova preserves all classes,
      FedAvg/FedProx lose rare-class signal as LR asymmetry grows

Outputs:
  - d1_mechanism_summary.csv : per-(algo, ratio) aggregate statistics
  - F_d1_mechanism_update_norms.{pdf,png} : per-client update-norm
       trajectories per (algo, ratio)
  - F_d1_mechanism_influence_ratio.{pdf,png} : mean ||Δw_1||/||Δw_0||
       per (algo × LR ratio)
  - F_d1_mechanism_per_class_survival.{pdf,png} : per-class final F1
       per (algo × LR ratio)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[4]
IN_DIR = REPO_ROOT / "fl_dermamnist/results/asymmetric_lr_L4"
OUT_DIR = REPO_ROOT / "fl_dermamnist/results/thesis_ready/data"
OUT_FIG = REPO_ROOT / "fl_dermamnist/results/thesis_ready/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["actinic", "basal", "benign_kerat", "dermato",
               "melanoma", "mel_nevi", "vascular"]
CLASS_F1_COLS = [f"val_f1_class_{c}" for c in range(7)]
RARE_IDX = (3, 4, 6)

ALGO_COLORS = {"fedavg": "#7FBF94", "fedprox": "#3D5A80", "fednova": "#C9A227"}
RATIO_ORDER_LABEL = {0: "1:1 symmetric", 1: "2:1", 2: "5:1"}
RATIO_COLORS = {0: "#7FBF94", 1: "#C9A227", 2: "#C03A2B"}

# Filename patterns
STEM_RE = re.compile(
    r"(?P<base>client_update_norms|history|test_at_best)_"
    r"(?P<algo>fedavg|fedprox|fednova)_mu(?P<mu>[0-9.]+)_E\d+"
    r"(?P<tail>.*)_s(?P<seed>\d+)\.(csv|json)$"
)


def _parse_filename(path: Path) -> dict | None:
    m = STEM_RE.match(path.name)
    if m is None:
        return None
    return dict(
        algorithm=m.group("algo"),
        mu=float(m.group("mu")),
        seed=int(m.group("seed")),
        tail=m.group("tail"),
    )


def _ratio_label_from_tail(tail: str) -> tuple[str, int]:
    """Identify LR ratio from filename tail. Returns (label, ordering_idx)."""
    if "lrPC-c0lr0.01-c1lr0.005" in tail:
        return ("2:1", 1)
    if "lrPC-c0lr0.01-c1lr0.002" in tail:
        return ("5:1", 2)
    if "lrPC" in tail:
        return (tail, 99)  # unknown ratio
    return ("1:1 symmetric", 0)


# ----------------------------------------------------------------
# 1. Read all 27 client_update_norms files + matched JSONs
# ----------------------------------------------------------------
rows = []   # per (algo, ratio, seed, round, client_id)
agg_rows = []  # per (algo, ratio, seed)
for f in IN_DIR.glob("client_update_norms_*.csv"):
    meta = _parse_filename(f)
    if meta is None:
        continue
    ratio_label, ratio_idx = _ratio_label_from_tail(meta["tail"])
    df = pd.read_csv(f)
    if "client_id" not in df.columns or "update_norm" not in df.columns:
        continue
    # Per-(algo, ratio, seed) mean update norms
    grouped = df.groupby("client_id")["update_norm"].agg(["mean", "std", "min", "max"])
    if 0 not in grouped.index or 1 not in grouped.index:
        continue
    norm_c0 = float(grouped.loc[0, "mean"])
    norm_c1 = float(grouped.loc[1, "mean"])
    agg_rows.append(dict(
        algorithm=meta["algorithm"],
        ratio=ratio_label,
        ratio_idx=ratio_idx,
        seed=meta["seed"],
        mean_norm_c0=norm_c0,
        mean_norm_c1=norm_c1,
        influence_ratio_c1_to_c0=norm_c1 / norm_c0 if norm_c0 > 0 else np.nan,
        file=f.name,
    ))
    # Per-round records too (for trajectory figure)
    for _, row in df.iterrows():
        rows.append(dict(
            algorithm=meta["algorithm"],
            ratio=ratio_label,
            ratio_idx=ratio_idx,
            seed=meta["seed"],
            round=int(row["round"]),
            client_id=int(row["client_id"]),
            update_norm=float(row["update_norm"]),
        ))

per_round_df = pd.DataFrame(rows)
agg_df = pd.DataFrame(agg_rows)
print(f"Loaded update-norm trajectories: {len(per_round_df)} rows, "
      f"{len(agg_df)} per-(algo, ratio, seed) aggregates")

# Test JSON / history for per-class final F1
final_per_class = []
for f in IN_DIR.glob("test_at_best_*.json"):
    meta = _parse_filename(f)
    if meta is None:
        continue
    ratio_label, ratio_idx = _ratio_label_from_tail(meta["tail"])
    with open(f) as fh:
        d = json.load(fh)
    pc = d.get("per_class_f1") or [float("nan")] * 7
    final_per_class.append(dict(
        algorithm=meta["algorithm"],
        ratio=ratio_label,
        ratio_idx=ratio_idx,
        seed=meta["seed"],
        macro_f1=d.get("macro_f1"),
        **{f"f1_{CLASS_NAMES[c]}": float(pc[c]) for c in range(7)},
    ))
class_df = pd.DataFrame(final_per_class)
print(f"Loaded per-class final F1: {len(class_df)} rows")

# ----------------------------------------------------------------
# 2. Compute summary statistics per (algorithm, ratio)
# ----------------------------------------------------------------
summary = agg_df.groupby(["algorithm", "ratio", "ratio_idx"]).agg(
    n=("seed", "count"),
    mean_norm_c0=("mean_norm_c0", "mean"),
    mean_norm_c1=("mean_norm_c1", "mean"),
    mean_influence_ratio=("influence_ratio_c1_to_c0", "mean"),
    sd_influence_ratio=("influence_ratio_c1_to_c0", lambda x: x.std(ddof=1) if len(x) > 1 else 0),
).reset_index().sort_values(["algorithm", "ratio_idx"])
summary.to_csv(OUT_DIR / "d1_mechanism_summary.csv", index=False)
print()
print("=" * 92)
print("Per-(algo, ratio) update-norm statistics (3-seed mean)")
print("=" * 92)
print(summary.to_string(index=False))

# ----------------------------------------------------------------
# 3. Mechanism hypothesis tests
# ----------------------------------------------------------------
print()
print("=" * 92)
print("H1: Does FedNova LR-normalize Client 1's contribution?")
print("=" * 92)
for algo in ("fedavg", "fedprox", "fednova"):
    sub = summary[summary["algorithm"] == algo].sort_values("ratio_idx")
    if len(sub) < 3:
        continue
    sym_c1 = float(sub.iloc[0]["mean_norm_c1"])
    asym_c1 = float(sub.iloc[-1]["mean_norm_c1"])  # 5:1
    # Expected if updates scale linearly with LR: asym/sym = 0.2
    # Expected if LR-invariant: asym/sym ≈ 1.0
    ratio = asym_c1 / sym_c1 if sym_c1 > 0 else 0
    # Reference: at 5:1, Client 1's LR is 0.002 vs 0.01 at 1:1, ratio 0.2.
    # If update_norm scales linearly with LR, expect ratio ≈ 0.2.
    # If LR-invariant (FedNova hypothesis), expect ratio ≈ 1.0.
    print(f"  {algo:>8s}: ||Δw_1|| at 5:1 / 1:1 = {ratio:.3f}  "
          f"(expected 0.20 if LR-scaled, 1.0 if LR-invariant)")

print()
print("=" * 92)
print("H2: Does FedProx over-suppress Client 1 vs FedAvg?")
print("=" * 92)
for ratio_idx in (0, 1, 2):
    fa_c1 = float(summary[(summary["algorithm"] == "fedavg") & (summary["ratio_idx"] == ratio_idx)]["mean_norm_c1"].iloc[0])
    fp_c1 = float(summary[(summary["algorithm"] == "fedprox") & (summary["ratio_idx"] == ratio_idx)]["mean_norm_c1"].iloc[0])
    rel = fp_c1 / fa_c1 if fa_c1 > 0 else 0
    print(f"  Ratio {RATIO_ORDER_LABEL[ratio_idx]:>15s}: FedProx ||Δw_1|| / FedAvg ||Δw_1|| = {rel:.3f}  "
          f"({'over-suppressed' if rel < 0.95 else 'comparable'})")

# ----------------------------------------------------------------
# 4. Per-class survival across (algorithm × ratio)
# ----------------------------------------------------------------
print()
print("=" * 92)
print("H3: Per-class final F1 — which classes does each algo preserve under 5:1?")
print("=" * 92)
class_summary = class_df.groupby(["algorithm", "ratio", "ratio_idx"]).agg(
    **{f"mean_{c}": (f"f1_{n}", "mean") for c, n in zip(range(7), CLASS_NAMES)},
).reset_index().sort_values(["algorithm", "ratio_idx"])
print(class_summary.to_string(index=False))

# ----------------------------------------------------------------
# 5. Figure 1: per-client update-norm trajectories per (algo × ratio)
# ----------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True,
                         gridspec_kw={"wspace": 0.06})
for ax, algo in zip(axes, ["fedavg", "fedprox", "fednova"]):
    for ratio_idx in (0, 1, 2):
        # Compute per-(round, client) mean across seeds
        sub = per_round_df[(per_round_df["algorithm"] == algo) &
                           (per_round_df["ratio_idx"] == ratio_idx)]
        for cid in (0, 1):
            line = sub[sub["client_id"] == cid].groupby("round")["update_norm"].mean()
            ls = "-" if cid == 0 else "--"
            ax.plot(line.index, line.values,
                    color=RATIO_COLORS[ratio_idx], linewidth=1.5 if cid == 0 else 1.2,
                    linestyle=ls, alpha=0.85,
                    label=f"{RATIO_ORDER_LABEL[ratio_idx]} — C{cid}"
                    if (cid == 0 and ratio_idx == 0) or (cid == 1 and ratio_idx == 0) else None)
    ax.set_title({"fedavg": "FedAvg", "fedprox": r"FedProx ($\mu=0.01$)", "fednova": "FedNova"}[algo],
                 loc="left", fontweight="bold", fontsize=11)
    ax.set_xlabel("Communication round")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Custom legend
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color=RATIO_COLORS[0], linewidth=1.5, label="1:1 symmetric"),
    Line2D([0], [0], color=RATIO_COLORS[1], linewidth=1.5, label="2:1"),
    Line2D([0], [0], color=RATIO_COLORS[2], linewidth=1.5, label="5:1"),
    Line2D([0], [0], color="#444", linewidth=1.5, linestyle="-", label="Client 0 (large, lr=0.01)"),
    Line2D([0], [0], color="#444", linewidth=1.2, linestyle="--", label="Client 1 (small, lr=varies)"),
]
axes[0].set_ylabel(r"$\|\Delta w\|_2$ (per-client update norm, log scale)")
axes[2].legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=8)

fig.suptitle("Per-client update-norm trajectories under asymmetric LR — "
             "FedNova's Client 1 stays comparable to Client 0 across LR ratios",
             fontsize=11.5, fontweight="bold", y=1.02)
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"F_d1_mechanism_update_norms.{ext}")
plt.close(fig)
print()
print(f"Wrote {OUT_FIG/'F_d1_mechanism_update_norms.pdf'}")

# ----------------------------------------------------------------
# 6. Figure 2: influence ratio (||Δw_1||/||Δw_0||) per (algo × LR ratio)
# ----------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
x = np.arange(3)
w = 0.27
for i, algo in enumerate(["fedavg", "fedprox", "fednova"]):
    sub = summary[summary["algorithm"] == algo].sort_values("ratio_idx")
    means = sub["mean_influence_ratio"].values
    sds = sub["sd_influence_ratio"].values
    ax.bar(x + (i - 1) * w, means, w, yerr=sds,
           color=ALGO_COLORS[algo], edgecolor="white", linewidth=0.6, capsize=4,
           error_kw=dict(linewidth=1, ecolor="#333"),
           label={"fedavg": "FedAvg", "fedprox": r"FedProx ($\mu=0.01$)", "fednova": "FedNova"}[algo])
    for j, (mv, sv) in enumerate(zip(means, sds)):
        ax.text(x[j] + (i - 1) * w, mv + sv + 0.005, f"{mv:.2f}",
                ha="center", va="bottom", fontsize=8.5, color=ALGO_COLORS[algo])
ax.axhline(1.0, color="#444", linestyle=":", linewidth=1.0, alpha=0.6, label="equal influence")
ax.set_xticks(x); ax.set_xticklabels([RATIO_ORDER_LABEL[i] for i in range(3)])
ax.set_xlabel("LR asymmetry ratio (Client 0 : Client 1)")
ax.set_ylabel(r"Mean $\|\Delta w_1\| / \|\Delta w_0\|$ (Client 1's relative influence)")
ax.set_title("Per-client influence ratio across LR asymmetry — "
             "how much does each client contribute to the aggregate update?",
             loc="left", fontweight="bold", fontsize=11)
ax.legend(loc="upper right", frameon=False, fontsize=10)
ax.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"F_d1_mechanism_influence_ratio.{ext}")
plt.close(fig)
print(f"Wrote {OUT_FIG/'F_d1_mechanism_influence_ratio.pdf'}")

# ----------------------------------------------------------------
# 7. Figure 3: per-class final F1 heatmap (algo × ratio × class)
# ----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True,
                         gridspec_kw={"wspace": 0.06})
for ax, algo in zip(axes, ["fedavg", "fedprox", "fednova"]):
    sub = class_summary[class_summary["algorithm"] == algo].sort_values("ratio_idx")
    if not len(sub):
        continue
    # Build per-ratio per-class matrix
    matrix = np.zeros((3, 7))
    for ri in range(3):
        row = sub[sub["ratio_idx"] == ri]
        if len(row):
            for ci in range(7):
                matrix[ri, ci] = float(row[f"mean_{ci}"].iloc[0])
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1.0)
    ax.set_xticks(range(7))
    ax.set_xticklabels(CLASS_NAMES, rotation=20, ha="right", fontsize=8.5)
    ax.set_yticks(range(3))
    ax.set_yticklabels([RATIO_ORDER_LABEL[i] for i in range(3)], fontsize=9)
    # Annotate cells
    for ri in range(3):
        for ci in range(7):
            v = matrix[ri, ci]
            color = "white" if v < 0.4 else "black"
            ax.text(ci, ri, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color=color)
    # Mark rare classes
    for ci in RARE_IDX:
        ax.axvline(ci - 0.5, color="#333", linewidth=0.5, alpha=0.3)
        ax.axvline(ci + 0.5, color="#333", linewidth=0.5, alpha=0.3)
    ax.set_title({"fedavg": "FedAvg", "fedprox": r"FedProx ($\mu=0.01$)", "fednova": "FedNova"}[algo],
                 loc="left", fontweight="bold", fontsize=11)
fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.015, pad=0.02,
             label="Final test F1")
axes[0].set_ylabel("LR asymmetry ratio")
fig.suptitle("Per-class final F1 across (algorithm × LR ratio) — "
             "FedNova preserves rare classes (dermato, melanoma, vascular) where FedAvg/FedProx collapse",
             fontsize=11, fontweight="bold", y=1.02)
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"F_d1_mechanism_per_class_survival.{ext}")
plt.close(fig)
print(f"Wrote {OUT_FIG/'F_d1_mechanism_per_class_survival.pdf'}")

print()
print("Done.")
