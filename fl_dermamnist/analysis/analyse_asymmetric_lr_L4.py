"""Experiment D1 -- Asymmetric LR analysis (NOVEL Li 2020 §5.2 extension).

Reads the asymmetric-LR sweep at
  fl_dermamnist/results/asymmetric_lr_L4/
      test_at_best_{fedavg,fedprox,fednova}_..._lrPC-c0lr0.01-c1lr{...}_s{42,123,456}.json
      test_at_best_{fedavg,fedprox,fednova}_mu0.X_E20_s{42,123,456}.json  (LR=1:1 baseline)

Six LR-asymmetry ratios (Client 0 lr / Client 1 lr) x 3 algorithms x 3 seeds = 54
runs total: 1:1, 2:1, 5:1, 10:1, 20:1, 50:1.

Tests whether the FedProx-vs-FedNova mechanism distinction extends to asymmetric
LR -- a regime where Wang 2020's FedNova theory does NOT apply (FedNova proves
correction for unequal tau_i, i.e. local work, not for learning rate).

Output:
  - asymmetric_lr_L4_summary.csv
  - F_asymmetric_lr_L4.{pdf,png}  : 2-panel line plot (macro-F1 + rare-class F1)
                                     across the six ratios, one line per algorithm

SAFETY: this script will refuse to overwrite the thesis CSV/figure unless all six
expected ratios and the full 54-run count are present, so a partial result set can
never silently clobber the committed six-ratio figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fl_dermamnist.common.paths import repo_root  # noqa: E402

REPO_ROOT = repo_root()
IN_DIR = REPO_ROOT / "fl_dermamnist/results/asymmetric_lr_L4"
OUT_DIR = REPO_ROOT / "fl_dermamnist/results/thesis_ready/data"
OUT_FIG = REPO_ROOT / "fl_dermamnist/results/thesis_ready/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["actinic", "basal", "benign_kerat", "dermato",
               "melanoma", "mel_nevi", "vascular"]
RARE_IDX = (3, 4, 6)

# Expected thesis design -- the safety gate checks the data against these.
EXPECTED_RATIOS = [1, 2, 5, 10, 20, 50]      # Client 0 lr / Client 1 lr
EXPECTED_RUNS = 54                            # 6 ratios x 3 algos x 3 seeds


def _ratio_label(r: int) -> str:
    return "1:1 (symmetric)" if r == 1 else f"{r}:1"


def _scan() -> pd.DataFrame:
    """Read all JSONs and parse the LR ratio from the lr_per_client metadata."""
    rows = []
    for f in sorted(IN_DIR.glob("test_at_best_*.json")):
        with open(f) as fh:
            d = json.load(fh)
        lr_pc = d.get("lr_per_client")
        global_lr = d.get("lr")
        if isinstance(lr_pc, dict):
            c0_lr = float(lr_pc.get("0", global_lr))
            c1_lr = float(lr_pc.get("1", global_lr))
        else:
            c0_lr = c1_lr = float(global_lr)
        ratio = int(round(c0_lr / c1_lr))          # 1, 2, 5, 10, 20, 50
        pc = d.get("per_class_f1") or [float("nan")] * 7
        rows.append(dict(
            algorithm=d.get("algorithm"),
            seed=d.get("seed"),
            c0_lr=c0_lr, c1_lr=c1_lr,
            ratio_value=ratio,
            ratio_label=_ratio_label(ratio),
            ratio_order=ratio,                      # numeric order == the ratio
            macro_f1=d.get("macro_f1"),
            balanced_accuracy=d.get("balanced_accuracy"),
            rare_avg_f1=float(np.mean([pc[i] for i in RARE_IDX])),
            **{f"f1_{CLASS_NAMES[i]}": pc[i] for i in range(7)},
            file=f.name,
        ))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 1. Load + SAFETY GATE
# ----------------------------------------------------------------------
df = _scan()
if df.empty:
    print(f"WARNING: no result files in {IN_DIR}. Submit jobs first.")
    raise SystemExit(0)
df = df.sort_values(["algorithm", "ratio_order", "seed"])

found_ratios = sorted(df["ratio_value"].unique().tolist())
SAFE = (len(df) == EXPECTED_RUNS) and (found_ratios == EXPECTED_RATIOS)
if not SAFE:
    print("=" * 72)
    print("SAFETY GATE: data does not match the expected six-ratio thesis design.")
    print(f"  expected {EXPECTED_RUNS} runs across ratios {EXPECTED_RATIOS}")
    print(f"  found    {len(df)} runs across ratios {found_ratios}")
    print("  Refusing to overwrite asymmetric_lr_L4_summary.csv or "
          "F_asymmetric_lr_L4.* .")
    print("  (Re-run once all runs are present.) Printing summary only.")
    print("=" * 72)

# ----------------------------------------------------------------------
# 2. 3-seed mean +/- SD per (algorithm, ratio)
# ----------------------------------------------------------------------
summary = df.groupby(["algorithm", "ratio_label", "ratio_order"]).agg(
    n=("macro_f1", "count"),
    mean_macro=("macro_f1", "mean"),
    sd_macro=("macro_f1", lambda x: x.std(ddof=1) if len(x) > 1 else 0.0),
    mean_rare=("rare_avg_f1", "mean"),
    sd_rare=("rare_avg_f1", lambda x: x.std(ddof=1) if len(x) > 1 else 0.0),
).reset_index().sort_values(["algorithm", "ratio_order"])
print("3-seed mean +/- SD per (algorithm, LR ratio):")
print(summary.to_string(index=False))

# ----------------------------------------------------------------------
# 3. LR-asymmetry penalty per algorithm: 1:1 -> most extreme (50:1)
# ----------------------------------------------------------------------
print("\nLR-asymmetry penalty: drop from symmetric (1:1) to most asymmetric "
      f"({EXPECTED_RATIOS[-1]}:1)")
for algo in ["fedavg", "fedprox", "fednova"]:
    sub = summary[summary["algorithm"] == algo]
    sym = sub[sub["ratio_order"] == 1]
    asy = sub[sub["ratio_order"] == EXPECTED_RATIOS[-1]]
    if not len(sym) or not len(asy):
        print(f"  {algo:<10s}: insufficient data")
        continue
    drop = float(sym["mean_macro"].iloc[0] - asy["mean_macro"].iloc[0])
    rare_drop = float(sym["mean_rare"].iloc[0] - asy["mean_rare"].iloc[0])
    tag = "absorbs" if drop < 0.02 else ("affected" if drop < 0.05 else "COLLAPSES")
    print(f"  {algo:<10s}: macro-F1 drop = {drop:+.4f}  rare drop = {rare_drop:+.4f}  --> {tag}")

if not SAFE:
    raise SystemExit(1)

df.to_csv(OUT_DIR / "asymmetric_lr_L4_summary.csv", index=False)
print(f"\nWrote {OUT_DIR/'asymmetric_lr_L4_summary.csv'}  ({len(df)} runs)")

# ----------------------------------------------------------------------
# 4. Figure: 2-panel line plot across the six ratios
# ----------------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "pdf.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight"})
COLORS = {"fedavg": "#C0392B", "fedprox": "#E67E22", "fednova": "#2E8B57"}
LABELS = {"fedavg": "FedAvg", "fedprox": r"FedProx ($\mu=0.01$)", "fednova": "FedNova"}
ORDER = EXPECTED_RATIOS
xticklabels = [_ratio_label(r).replace(" (symmetric)", "") for r in ORDER]
x = np.arange(len(ORDER))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.0, 5.0),
                               gridspec_kw={"wspace": 0.22})


def _series(sub, col):
    means, sds = [], []
    for r in ORDER:
        s = sub[sub["ratio_order"] == r]
        means.append(float(s[f"mean_{col}"].iloc[0]) if len(s) else np.nan)
        sds.append(float(s[f"sd_{col}"].iloc[0]) if len(s) else 0.0)
    return np.array(means), np.array(sds)


for col, ax, ylab, title in [
    ("macro", axA, "test macro-F1", "(a) Macro-F1 vs LR asymmetry"),
    ("rare", axB, "rare-class F1 (mean over derm/mel/vasc)", "(b) Rare-class F1 vs LR asymmetry"),
]:
    for algo in ["fedavg", "fedprox", "fednova"]:
        sub = summary[summary["algorithm"] == algo]
        m, sd = _series(sub, col)
        ax.errorbar(x, m, yerr=sd, marker="o", ms=5, lw=1.6, capsize=3,
                    color=COLORS[algo], label=LABELS[algo])
    ax.set_xticks(x); ax.set_xticklabels(xticklabels)
    ax.set_xlabel("Client 1 LR asymmetry ratio  (Client 0 lr / Client 1 lr)")
    ax.set_ylabel(ylab)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10.5)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
axA.legend(loc="lower left", frameon=False, fontsize=9)

for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"F_asymmetric_lr_L4.{ext}")
plt.close(fig)
print(f"Wrote {OUT_FIG/'F_asymmetric_lr_L4.pdf'}")
print("Done.")
