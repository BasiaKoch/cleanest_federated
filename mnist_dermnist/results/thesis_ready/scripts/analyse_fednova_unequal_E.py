"""Experiment 3 — FedNova × equal-vs-unequal local epochs analysis.

Reads the 12 runs at
  mnist_dermnist/results/fednova_unequal_E/L{3,4}_<partition>/
       test_at_best_{fedavg,fedprox,fednova}_..._E20[_sh-fixed_stragglers]_s42.json
and produces a 2 × 3 mechanism-decomposition table:

                  | equal-E (E_0=20, E_1=20) | unequal-E (E_0=20, E_1=5)
  ----------------+--------------------------+---------------------------
  FedAvg          |          ✓               |          ✓
  FedProx (μ=0.01)|          ✓               |          ✓
  FedNova         |          ✓               |          ✓

The headline mechanism-isolation reading (Wang 2020 §5 + NIID-Bench
§4.2 framing):
  - Compare the FedNova → FedAvg delta in equal-E (should be ≈ 0:
    FedNova reduces to FedAvg under equal τ_i) vs unequal-E (FedNova
    should win, that is the FedNova-paper claim).
  - Compare the FedProx → FedAvg delta in unequal-E: this is the
    "drift-damping under unequal work" effect, which FedProx provides
    only partially. The remaining FedNova-vs-FedProx gap in unequal-E
    is the OBJECTIVE-INCONSISTENCY component that FedProx cannot fix.

Outputs:
  - fednova_unequal_E_summary.csv  (long format, 12 rows)
  - F_fednova_unequal_E.{pdf,png}  (bar chart: 2 levels × 3 algos × 2 regimes)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[4]
IN_ROOT = REPO_ROOT / "mnist_dermnist/results/fednova_unequal_E"
OUT_DIR = IN_ROOT / "analysis"
OUT_FIG = REPO_ROOT / "mnist_dermnist/results/thesis_ready/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

LEVELS = [
    ("L3", "two_client_70_30_rare_enriched"),
    ("L4", "two_client_90_10_rare_stress"),
]
ALGOS = ["fedavg", "fedprox", "fednova"]
MU_BY_ALGO = {"fedavg": "0.0", "fedprox": "0.01", "fednova": "0.0"}

CLASS_NAMES = [
    "actinic", "basal", "benign_kerat", "dermato", "melanoma", "mel_nevi", "vascular",
]
RARE_IDX = (3, 4, 6)


def _read_test_at_best(d: Path, algo: str, mu: str, sh_tag: str) -> dict | None:
    """sh_tag = '' for equal-E, '_sh-fixed_stragglers' for unequal-E."""
    f = d / f"test_at_best_{algo}_mu{mu}_E20{sh_tag}_s42.json"
    if not f.exists():
        return None
    return json.load(open(f))


rows = []
for level, partition in LEVELS:
    d = IN_ROOT / f"{level}_{partition}"
    for algo in ALGOS:
        mu = MU_BY_ALGO[algo]
        for regime, sh_tag in [("equal", ""), ("unequal", "_sh-fixed_stragglers")]:
            x = _read_test_at_best(d, algo, mu, sh_tag)
            if x is None:
                print(f"  WARN: missing {level}/{algo}/{regime}")
                continue
            pc = x.get("per_class_f1") or [float("nan")] * 7
            rows.append(dict(
                level=level, partition=partition,
                algorithm=algo, mu=float(mu), regime=regime,
                hostname=x.get("hostname", "?"),
                selected_round=x.get("selected_round"),
                macro_f1=x.get("macro_f1"),
                balanced_accuracy=x.get("balanced_accuracy"),
                rare_avg_f1=float(np.mean([pc[i] for i in RARE_IDX])),
                **{f"f1_{CLASS_NAMES[i]}": pc[i] for i in range(7)},
            ))
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "fednova_unequal_E_summary.csv", index=False)
print(f"Wrote {OUT_DIR / 'fednova_unequal_E_summary.csv'}  ({len(df)} runs)")
print()
print(df[["level", "algorithm", "regime", "selected_round",
         "macro_f1", "rare_avg_f1", "f1_vascular"]].to_string(index=False))

# --- Mechanism decomposition reading ---
print()
print("=" * 78)
print("Mechanism decomposition (per level)")
print("=" * 78)
for level, _ in LEVELS:
    sub = df[df["level"] == level]
    if not len(sub):
        continue
    print()
    print(f"--- {level} ---")
    pivot = sub.pivot_table(index="algorithm", columns="regime", values="macro_f1")
    pivot = pivot.reindex(index=ALGOS, columns=["equal", "unequal"])
    print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))
    print()
    # The mechanism-isolation deltas:
    if "fedavg" in pivot.index and "fednova" in pivot.index:
        fnv_eq    = pivot.loc["fednova", "equal"]    if "equal"   in pivot.columns else np.nan
        fnv_uneq  = pivot.loc["fednova", "unequal"]  if "unequal" in pivot.columns else np.nan
        fa_eq     = pivot.loc["fedavg",  "equal"]    if "equal"   in pivot.columns else np.nan
        fa_uneq   = pivot.loc["fedavg",  "unequal"]  if "unequal" in pivot.columns else np.nan
        fp_uneq   = pivot.loc["fedprox", "unequal"]  if ("fedprox" in pivot.index and "unequal" in pivot.columns) else np.nan
        d_eq      = fnv_eq   - fa_eq
        d_uneq    = fnv_uneq - fa_uneq
        d_diff    = d_uneq  - d_eq
        print(f"  FedNova − FedAvg | equal-E   = {d_eq:+.4f}  (should be ≈ 0; FedNova reduces to FedAvg under equal τ_i)")
        print(f"  FedNova − FedAvg | unequal-E = {d_uneq:+.4f}  (objective-inconsistency correction; Wang 2020 prediction)")
        print(f"  ΔΔ (mechanism slope)         = {d_diff:+.4f}  (the part FedNova adds that FedProx cannot)")
        if not np.isnan(fp_uneq):
            d_residual = fnv_uneq - fp_uneq
            print(f"  FedNova − FedProx | unequal-E = {d_residual:+.4f}  (objective-inconsistency residual after drift damping)")

# --- Figure: 2-level grouped bar chart ---
COLORS = {"fedavg": "#7FBF94", "fedprox": "#3D5A80", "fednova": "#C9A227"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5),
                         gridspec_kw={"wspace": 0.18}, sharey=True)

for ax, (level, _) in zip(axes, LEVELS):
    sub = df[df["level"] == level]
    regimes = ["equal", "unequal"]
    x = np.arange(2)
    w = 0.27
    for i, algo in enumerate(ALGOS):
        vals = []
        for reg in regimes:
            s = sub[(sub["algorithm"] == algo) & (sub["regime"] == reg)]
            vals.append(float(s["macro_f1"].iloc[0]) if len(s) else np.nan)
        bars = ax.bar(x + (i - 1) * w, vals, w,
                      color=COLORS[algo], edgecolor="white", linewidth=0.6,
                      label={"fedavg":"FedAvg","fedprox":r"FedProx ($\mu=0.01$)","fednova":"FedNova"}[algo])
        for j, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(x[j] + (i - 1) * w, v + 0.005, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8.5,
                        color=COLORS[algo], fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([
        "equal-E\n($E_0=E_1=20$)",
        "unequal-E\n($E_0=20, E_1=5$)"])
    ax.set_title(f"{level}: {LEVELS[0][1] if level == 'L3' else LEVELS[1][1]}",
                 loc="left", fontweight="bold", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Test macro-F1 (seed 42)")
axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, 1.20),
               ncol=3, frameon=False, fontsize=10)
fig.suptitle("FedNova × equal-vs-unequal local epochs — drift-damping vs objective-inconsistency decomposition",
             fontsize=11, fontweight="bold", y=1.04)
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"F_fednova_unequal_E.{ext}")
plt.close(fig)
print()
print(f"Wrote {OUT_FIG / 'F_fednova_unequal_E.pdf'}")
print()
print("Done.")
