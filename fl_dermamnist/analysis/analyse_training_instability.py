"""Experiment C1 - Training-instability replication on medical FL.

Replicates the training-instability framework of Charles et al. 2024
("Not All FL Algorithms Are Created Equal," arXiv:2403.17287) on
DermaMNIST, which their paper does NOT evaluate. They define training
instability as the std of test accuracy over the final 10 rounds of
training. We compute the same metric on val macro-F1 (test is only
available at the best-val checkpoint in our setup) across all our
existing runs.

Why this is a contribution:
  - Charles et al. 2024 §3.1 identifies training stability as
    under-reported on medical-FL with class imbalance
  - Their evaluation is on CIFAR/FEMNIST with ≥10 clients
  - The 2-client medical-FL case is absent from their table
  - We extend their cross-method instability comparison to medical data

Expected to replicate two of their findings on DermaMNIST:
  1. FedProx is MORE stable than FedAvg under symmetric protocol
  2. FedNova is LESS stable than FedAvg under unequal-E

Output:
  - training_instability_summary.csv : per-run + per-(algo,partition) std
  - F_training_instability.{pdf,png}  : 2-panel grouped bar chart
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
RESULTS = REPO_ROOT / "fl_dermamnist/results"
OUT_DIR = REPO_ROOT / "fl_dermamnist/results/thesis_ready/data"
OUT_FIG = REPO_ROOT / "fl_dermamnist/results/thesis_ready/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

K = 10  # window for instability (matches Charles 2024 §3.2)


# ----------------------------------------------------------------
# Experiment scan registry - which directories define which "regime"
# (symmetric vs unequal-E)
# ----------------------------------------------------------------
SOURCES = [
    # (regime_label, partition_label, root_dir, recursive)
    ("symmetric L4 (node_pinned)",      "L4",                RESULTS / "node_pinned_L4",           False),
    ("symmetric L4 (90/10 baseline)",   "L4",                RESULTS / "two_client_90_10_rare_stress", False),
    ("extended L3 (250r)",              "L3",                RESULTS / "extended_rounds_L3",       False),
    ("L4 asymmetric μ",                 "L4",                RESULTS / "asymmetric_mu_L4",         False),
    ("FedNova × E (L3 equal & unequal)", "L3",                RESULTS / "fednova_unequal_E/L3_two_client_70_30_rare_enriched", False),
    ("FedNova × E (L4 equal & unequal)", "L4",                RESULTS / "fednova_unequal_E/L4_two_client_90_10_rare_stress",     False),
    ("Li 2020 §5.2 protocol",           "L4",                RESULTS / "li2020_asymmetric_L4",     False),
    ("perfect-storm",                   "L4",                RESULTS / "fedprox_perfect_storm_L4", False),
    ("μ-sweep (L0)",                    "L0",                RESULTS / "mu_sweep_ladder/L0_two_client_50_50_stratified_iid", False),
    ("μ-sweep (L2)",                    "L2",                RESULTS / "mu_sweep_ladder/L2_two_client_50_50_label_skew_only", False),
    ("μ-sweep (L4)",                    "L4",                RESULTS / "mu_sweep_ladder/L4_two_client_90_10_rare_stress", False),
    ("heterogeneity ladder",            "varied",            RESULTS / "heterogeneity_ladder",     True),
]


STEM_RE = re.compile(
    r"history_(?P<algo>fedavg|fedprox|fednova)_mu(?P<mu>[0-9.]+)_E\d+(?P<tail>.*)_s(?P<seed>\d+)\.csv"
)


def _parse(path: Path) -> dict | None:
    m = STEM_RE.match(path.name)
    if m is None:
        return None
    return dict(
        algorithm=m.group("algo"),
        mu=float(m.group("mu")),
        seed=int(m.group("seed")),
        tail=m.group("tail"),
    )


def _is_unequal_E(tail: str) -> bool:
    """Identify runs that used unequal-E (stragglers)."""
    return "sh-fixed_stragglers" in tail or "sh-random_stragglers" in tail


def _is_drop(tail: str) -> bool:
    return "_drop" in tail


def _instability(df: pd.DataFrame, k: int = K) -> dict:
    """std and mean of val_macro_f1 over the last k rounds."""
    if "val_macro_f1" not in df.columns or len(df) < 2:
        return dict(std_last_K=np.nan, mean_last_K=np.nan, range_last_K=np.nan, n_rounds=len(df))
    s = df["val_macro_f1"].dropna().iloc[-k:]
    return dict(
        std_last_K=float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        mean_last_K=float(s.mean()),
        range_last_K=float(s.max() - s.min()) if len(s) > 1 else 0.0,
        n_rounds=len(df),
    )


# ----------------------------------------------------------------
# 1. Scan all CSVs and compute per-run instability
# ----------------------------------------------------------------
rows = []
for source_label, partition_label, root, recursive in SOURCES:
    if not root.exists():
        continue
    iterator = root.rglob("history_*.csv") if recursive else root.glob("history_*.csv")
    for f in iterator:
        meta = _parse(f)
        if meta is None:
            continue
        df = pd.read_csv(f)
        inst = _instability(df)
        rows.append(dict(
            source=source_label,
            partition=partition_label,
            algorithm=meta["algorithm"],
            mu=meta["mu"],
            seed=meta["seed"],
            unequal_E=_is_unequal_E(meta["tail"]),
            drop_stragglers=_is_drop(meta["tail"]),
            file=f.name,
            **inst,
        ))
df_runs = pd.DataFrame(rows)
df_runs.to_csv(OUT_DIR / "training_instability_runs.csv", index=False)
print(f"Wrote {OUT_DIR/'training_instability_runs.csv'}  ({len(df_runs)} runs)")

# ----------------------------------------------------------------
# 2. Headline 1: Symmetric protocol (no stragglers, no drop) by algorithm
# ----------------------------------------------------------------
print()
print("=" * 80)
print("HEADLINE 1: Training instability under SYMMETRIC protocol")
print("(Replicates Charles 2024 Finding: FedProx more stable than FedAvg)")
print("=" * 80)
sym = df_runs[(~df_runs["unequal_E"]) & (~df_runs["drop_stragglers"])]
sym_summary = sym.groupby(["partition", "algorithm"]).agg(
    n=("std_last_K", "count"),
    mean_std=("std_last_K", "mean"),
    median_std=("std_last_K", "median"),
).reset_index()
print(sym_summary.to_string(index=False))

# ----------------------------------------------------------------
# 3. Headline 2: Unequal-E protocol - FedNova stability
# ----------------------------------------------------------------
print()
print("=" * 80)
print("HEADLINE 2: Training instability under UNEQUAL-E (FedNova target regime)")
print("(Replicates Charles 2024 Takeaway 10: 'FedNova is unstable')")
print("=" * 80)
ueE = df_runs[(df_runs["unequal_E"]) & (~df_runs["drop_stragglers"])]
ueE_summary = ueE.groupby(["partition", "algorithm"]).agg(
    n=("std_last_K", "count"),
    mean_std=("std_last_K", "mean"),
    median_std=("std_last_K", "median"),
).reset_index()
print(ueE_summary.to_string(index=False))

# ----------------------------------------------------------------
# 4. Pivot (algo × regime) showing instability
# ----------------------------------------------------------------
print()
print("=" * 80)
print("CONSOLIDATED: mean std_last_K by (algorithm × regime)")
print("=" * 80)
df_runs["regime"] = df_runs.apply(
    lambda r: "symmetric" if not r["unequal_E"] and not r["drop_stragglers"]
    else ("unequal_E (γ-inexact)" if r["unequal_E"] and not r["drop_stragglers"]
          else "drop-stragglers" if r["drop_stragglers"]
          else "other"),
    axis=1,
)
pivot = df_runs.groupby(["algorithm", "regime"]).agg(
    n=("std_last_K", "count"),
    mean_std=("std_last_K", "mean"),
    sd_std=("std_last_K", "std"),
).reset_index()
pivot.to_csv(OUT_DIR / "training_instability_pivot.csv", index=False)
print(pivot.to_string(index=False))

# ----------------------------------------------------------------
# 5. Plot: 2-panel grouped bar chart
# ----------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.5),
                                gridspec_kw={"wspace": 0.25})

COLORS = {"fedavg": "#7FBF94", "fedprox": "#3D5A80", "fednova": "#C9A227"}

# Panel A: symmetric protocol - algorithm comparison per partition
sym_pivot = sym_summary.pivot(index="partition", columns="algorithm", values="mean_std").fillna(0)
partitions_A = sym_pivot.index.tolist()
algos_A = ["fedavg", "fedprox", "fednova"]
x = np.arange(len(partitions_A))
w = 0.27
for i, algo in enumerate(algos_A):
    vals = sym_pivot.get(algo, pd.Series(0, index=partitions_A)).reindex(partitions_A).fillna(0).values
    axA.bar(x + (i - 1) * w, vals, w, color=COLORS[algo],
            edgecolor="white", linewidth=0.6,
            label={"fedavg":"FedAvg","fedprox":r"FedProx ($\mu=0.01$)","fednova":"FedNova"}[algo])
    for j, v in enumerate(vals):
        if v > 0:
            axA.text(x[j] + (i - 1) * w, v + 0.001, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=8, color=COLORS[algo])
axA.set_xticks(x); axA.set_xticklabels(partitions_A, fontsize=9)
axA.set_ylabel(rf"Mean std of val macro-F1 over last {K} rounds")
axA.set_title("(a) Symmetric protocol — algorithm-level instability by partition",
              loc="left", fontweight="bold", fontsize=11)
axA.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)
axA.legend(loc="upper right", frameon=False, fontsize=9)

# Panel B: equal vs unequal-E for FedNova (key story)
fn_data = df_runs[(df_runs["algorithm"] == "fednova")]
if len(fn_data) > 0:
    eq = fn_data[~fn_data["unequal_E"]]["std_last_K"].mean()
    ueq = fn_data[fn_data["unequal_E"]]["std_last_K"].mean()
    fa_eq = df_runs[(df_runs["algorithm"] == "fedavg") & (~df_runs["unequal_E"]) & (~df_runs["drop_stragglers"])]["std_last_K"].mean()
    fa_ueq = df_runs[(df_runs["algorithm"] == "fedavg") & (df_runs["unequal_E"]) & (~df_runs["drop_stragglers"])]["std_last_K"].mean()
    fp_eq = df_runs[(df_runs["algorithm"] == "fedprox") & (~df_runs["unequal_E"]) & (~df_runs["drop_stragglers"])]["std_last_K"].mean()
    fp_ueq = df_runs[(df_runs["algorithm"] == "fedprox") & (df_runs["unequal_E"]) & (~df_runs["drop_stragglers"])]["std_last_K"].mean()

    groups = ["FedAvg", "FedProx", "FedNova"]
    eqs = [fa_eq, fp_eq, eq]
    ueqs = [fa_ueq, fp_ueq, ueq]
    xg = np.arange(len(groups))
    w2 = 0.35
    axB.bar(xg - w2/2, eqs, w2,
            color=[COLORS[a] for a in ["fedavg","fedprox","fednova"]],
            edgecolor="white", linewidth=0.6, label="equal-E")
    axB.bar(xg + w2/2, ueqs, w2,
            color=[COLORS[a] for a in ["fedavg","fedprox","fednova"]],
            edgecolor="black", linewidth=1.0, hatch="//", alpha=0.85,
            label="unequal-E (γ-inexact)")
    for j, (eq_v, ueq_v) in enumerate(zip(eqs, ueqs)):
        if not np.isnan(eq_v):
            axB.text(xg[j] - w2/2, eq_v + 0.001, f"{eq_v:.3f}",
                     ha="center", va="bottom", fontsize=8)
        if not np.isnan(ueq_v):
            axB.text(xg[j] + w2/2, ueq_v + 0.001, f"{ueq_v:.3f}",
                     ha="center", va="bottom", fontsize=8)
    axB.set_xticks(xg); axB.set_xticklabels(groups, fontsize=10)
    axB.set_ylabel(rf"Mean std of val macro-F1 over last {K} rounds")
    axB.set_title("(b) Equal vs unequal-E — FedNova destabilises in its target regime",
                  loc="left", fontweight="bold", fontsize=11)
    axB.legend(loc="upper left", frameon=False, fontsize=9)
    axB.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
    axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)

fig.suptitle("Training instability — replication of Charles et al. 2024 (arXiv:2403.17287) on DermaMNIST 2-client medical FL",
             fontsize=11.5, fontweight="bold", y=1.02)
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"F_training_instability.{ext}")
plt.close(fig)
print()
print(f"Wrote {OUT_FIG/'F_training_instability.pdf'}")
print()
print("Done.")
