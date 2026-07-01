"""Experiment A1 - B-local dissimilarity proxy from existing update-norm logs.

Li 2020 Theorem 4 (FedProx, MLSys) assumes bounded B-local dissimilarity:
   B²(w) = max_k ||∇F_k(w)||² / ||∇F(w)||²
Nobody has empirically measured B(t) across heterogeneity levels on a
medical-FL benchmark and correlated it with the FedProx win/loss
direction. Yuan & Li (arXiv:2206.05187, NeurIPS 2022) removed the
bounded-B assumption theoretically but did not measure it; FedImpro
(arXiv:2402.07011, 2024) and HAPI-FedProx (Springer 2025) introduced
related update-level diagnostics but on different datasets.

EXACT B-dissimilarity requires gradient computation at the global
anchor (one extra forward+backward pass per client per round); we
have NOT logged this. Adding it requires re-running. As a CHEAPER
proxy, we use the existing per-(round, client) update norms
   ||Δw_k(t)|| = ||w_k^{t+1} - w^t||₂
which already exist in client_update_norms_*.csv across all our
runs. For small learning rate and few local epochs, this is
proportional to ||∇F_k(w^t)||₂ - so the RATIO
   B_proxy²(t) = max_k ||Δw_k(t)||² / mean_k ||Δw_k(t)||²
is a sound HETEROGENEITY DIAGNOSTIC even though it is not Li 2020's
exact B.

Reported in the thesis as:
   "We use an update-norm-based heterogeneity proxy, computable from
    existing logs. The exact B-local dissimilarity of Li 2020 Theorem 4
    would require additional gradient logging; we leave that to future
    work."

Output:
  - b_dissimilarity_proxy_summary.csv
  - F_b_dissimilarity_proxy.{pdf,png}  : 2-panel: trajectory + correlation
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


# Source: μ-sweep ladder + heterogeneity ladder both have update-norm logs
LADDER = [
    ("L0", "two_client_50_50_stratified_iid", "IID 50/50",          0.0000),
    ("L1", "two_client_86_14_quantity_only_stratified", "Quantity 86/14", 0.0000),
    ("L2", "two_client_50_50_label_skew_only", "Label-skew 50/50",  0.1037),
    ("L3", "two_client_70_30_rare_enriched", "Mixed 70/30",         0.1206),
    ("L4", "two_client_90_10_rare_stress", "Severe 90/10",          0.3853),
]


def _b_proxy_trajectory(csv_path: Path) -> dict | None:
    """Compute per-round max/mean update-norm ratio across clients."""
    df = pd.read_csv(csv_path)
    if "client_id" not in df.columns or "update_norm" not in df.columns or "round" not in df.columns:
        return None
    pivot = df.pivot_table(index="round", columns="client_id", values="update_norm")
    # Per round: max-to-mean ratio
    max_per_round = pivot.max(axis=1)
    mean_per_round = pivot.mean(axis=1)
    ratio = max_per_round / mean_per_round.replace(0, np.nan)
    return dict(
        rounds=pivot.index.tolist(),
        max=max_per_round.tolist(),
        mean=mean_per_round.tolist(),
        ratio=ratio.tolist(),
        mean_ratio=float(ratio.mean()),
        mean_max=float(max_per_round.mean()),
        mean_avg=float(mean_per_round.mean()),
    )


# ----------------------------------------------------------------
# 1. Build the per-run table
# ----------------------------------------------------------------
rows = []
for level, partition, label, js in LADDER:
    d = RESULTS / f"heterogeneity_ladder/{level}_{partition}"
    if not d.exists():
        continue
    for f in d.glob("client_update_norms_*.csv"):
        m = re.match(r"client_update_norms_(?P<algo>fedavg|fedprox|fednova)_mu(?P<mu>[0-9.]+)_E\d+(?P<tail>.*)_s(?P<seed>\d+)\.csv", f.name)
        if m is None:
            continue
        traj = _b_proxy_trajectory(f)
        if traj is None:
            continue
        # Read corresponding test_at_best to get the final macro-F1
        json_path = f.parent / f.name.replace("client_update_norms_", "test_at_best_").replace(".csv", ".json")
        macro_f1 = None
        if json_path.exists():
            try:
                macro_f1 = float(json.load(open(json_path))["macro_f1"])
            except Exception:
                pass
        rows.append(dict(
            level=level, partition=partition, label=label, js_divergence=js,
            algorithm=m.group("algo"), mu=float(m.group("mu")),
            seed=int(m.group("seed")),
            mean_ratio=traj["mean_ratio"],
            mean_max=traj["mean_max"],
            mean_avg=traj["mean_avg"],
            macro_f1=macro_f1,
            file=f.name,
        ))
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "b_dissimilarity_proxy_summary.csv", index=False)
print(f"Wrote {OUT_DIR/'b_dissimilarity_proxy_summary.csv'}  ({len(df)} runs)")
print()
print(df[["level", "label", "js_divergence", "algorithm", "mu",
         "mean_ratio", "macro_f1"]].to_string(index=False))

# ----------------------------------------------------------------
# 2. Compute FedProx − FedAvg macro-F1 gap per level + mean B-proxy
# ----------------------------------------------------------------
gaps = []
for level, partition, label, js in LADDER:
    sub = df[df["level"] == level]
    fa = sub[sub["algorithm"] == "fedavg"]
    fp = sub[sub["algorithm"] == "fedprox"]
    if not len(fa) or not len(fp):
        continue
    fa_macro = float(fa["macro_f1"].iloc[0]) if not fa["macro_f1"].isna().all() else np.nan
    fp_macro = float(fp["macro_f1"].iloc[0]) if not fp["macro_f1"].isna().all() else np.nan
    fa_ratio = float(fa["mean_ratio"].iloc[0])
    fp_ratio = float(fp["mean_ratio"].iloc[0])
    gaps.append(dict(
        level=level, label=label, js_divergence=js,
        fedavg_macro=fa_macro, fedprox_macro=fp_macro,
        delta_macro_f1=fp_macro - fa_macro,
        fedavg_b_proxy=fa_ratio, fedprox_b_proxy=fp_ratio,
        mean_b_proxy=(fa_ratio + fp_ratio) / 2,
    ))
gaps_df = pd.DataFrame(gaps)
print()
print("=" * 80)
print("Per-level B-proxy vs (FedProx − FedAvg) macro-F1 gap")
print("=" * 80)
print(gaps_df.to_string(index=False))

# Spearman correlation
if len(gaps_df) >= 3:
    from scipy.stats import spearmanr, pearsonr
    sp_r, sp_p = spearmanr(gaps_df["mean_b_proxy"], gaps_df["delta_macro_f1"])
    pe_r, pe_p = pearsonr(gaps_df["mean_b_proxy"], gaps_df["delta_macro_f1"])
    print()
    print(f"Spearman r = {sp_r:+.3f}  (p = {sp_p:.3f})")
    print(f"Pearson  r = {pe_r:+.3f}  (p = {pe_p:.3f})")

# ----------------------------------------------------------------
# 3. Figure: 2-panel - B-proxy trajectories + correlation
# ----------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.5),
                                gridspec_kw={"wspace": 0.25})

# Panel A: B-proxy trajectories per level (FedAvg only, one curve per level)
LEVEL_COLORS = {"L0": "#7FBF94", "L1": "#88C0D0", "L2": "#3D5A80",
                "L3": "#C9A227", "L4": "#C03A2B"}

for level, partition, label, js in LADDER:
    csv_p = RESULTS / f"heterogeneity_ladder/{level}_{partition}/client_update_norms_fedavg_mu0.0_E20_s42.csv"
    if not csv_p.exists():
        continue
    traj = _b_proxy_trajectory(csv_p)
    if traj is None:
        continue
    axA.plot(traj["rounds"], traj["ratio"],
             color=LEVEL_COLORS.get(level, "#666"),
             linewidth=1.3, alpha=0.8,
             label=f"{level} ({label}, JS={js:.2f})")
axA.set_xlabel("Communication round")
axA.set_ylabel("B-proxy = max_k‖Δw_k‖ / mean_k‖Δw_k‖")
axA.set_title("(a) B-dissimilarity proxy trajectories — FedAvg, seed 42",
              loc="left", fontweight="bold", fontsize=11)
axA.legend(loc="upper right", frameon=False, fontsize=8)
axA.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)

# Panel B: correlation - mean B-proxy vs FedProx−FedAvg gap
for _, row in gaps_df.iterrows():
    color = LEVEL_COLORS.get(row["level"], "#666")
    axB.scatter(row["mean_b_proxy"], row["delta_macro_f1"],
                s=150, color=color, edgecolor="black", linewidth=1.2, zorder=3,
                label=row["level"])
    axB.annotate(row["level"], (row["mean_b_proxy"], row["delta_macro_f1"]),
                 xytext=(8, 4), textcoords="offset points", fontsize=9)
axB.axhline(0, color="#888", linewidth=0.8)
axB.set_xlabel("Mean B-proxy across rounds  (heterogeneity diagnostic)")
axB.set_ylabel(r"$\Delta$ macro-F1 (FedProx − FedAvg)")
axB.set_title(f"(b) Higher B → FedProx advantage? "
              f"(Spearman r = {sp_r:+.2f}, p = {sp_p:.2f})" if len(gaps_df) >= 3
              else "(b) Heterogeneity vs FedProx advantage",
              loc="left", fontweight="bold", fontsize=11)
axB.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)

fig.suptitle("B-dissimilarity proxy from update-norm logs — proxy for Li 2020 Theorem 4 quantity",
             fontsize=11.5, fontweight="bold", y=1.02)
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"F_b_dissimilarity_proxy.{ext}")
plt.close(fig)
print()
print(f"Wrote {OUT_FIG/'F_b_dissimilarity_proxy.pdf'}")
print()
print("Done.")
