"""Node-pinned 3-seed L4 — variance isolation analysis.

Reads the 6 runs at
  fl_dermamnist/results/node_pinned_L4/test_at_best_{fedavg,fedprox}_mu*_E20_s{42,123,456}.json

and decides whether the L4 FedAvg-vs-FedProx deficit is a real
algorithmic effect or cross-seed CUDA noise on a single node.

Decision rule (printed prominently at the end):
  - |Δ_means| <= 2 × pooled_SD  →  algorithms statistically indistinguishable
  - |Δ_means| >  2 × pooled_SD  →  real effect on this node

Emits one summary figure F_node_pinned_L4.{pdf,png}:
  - left panel:  dot-plot of all 6 (algo, seed) macro-F1 with means
  - right panel: per-class F1 mean ± SD bars, FedAvg vs FedProx
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[4]
IN_DIR = REPO_ROOT / "fl_dermamnist/results/node_pinned_L4"
OUT_DIR = IN_DIR / "analysis"
OUT_FIG = REPO_ROOT / "fl_dermamnist/results/thesis_ready/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]
ALGOS = [("fedavg", "0.0"), ("fedprox", "0.01")]
CLASS_NAMES = [
    "actinic", "basal", "benign_kerat", "dermato", "melanoma", "mel_nevi", "vascular",
]
RARE_IDX = (3, 4, 6)

rows = []
hostnames = set()
for algo, mu in ALGOS:
    for seed in SEEDS:
        f = IN_DIR / f"test_at_best_{algo}_mu{mu}_E20_s{seed}.json"
        if not f.exists():
            print(f"  WARN: missing {f}")
            continue
        x = json.load(open(f))
        pc = x.get("per_class_f1") or [float("nan")] * 7
        hostnames.add(x.get("hostname", "?"))
        rows.append(dict(
            algorithm=algo, mu=float(mu), seed=seed,
            hostname=x.get("hostname", "?"),
            selected_round=x.get("selected_round"),
            macro_f1=x.get("macro_f1"),
            balanced_accuracy=x.get("balanced_accuracy"),
            rare_avg_f1=float(np.mean([pc[i] for i in RARE_IDX])),
            **{f"f1_{CLASS_NAMES[i]}": pc[i] for i in range(7)},
        ))
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "node_pinned_L4_summary.csv", index=False)
print(f"Wrote {OUT_DIR / 'node_pinned_L4_summary.csv'}  ({len(df)} runs)")
print()
print(df[["algorithm", "seed", "hostname", "selected_round",
         "macro_f1", "balanced_accuracy", "rare_avg_f1"]].to_string(index=False))
print()
print(f"Unique hostnames across all 6 runs: {sorted(hostnames)}")
if len(hostnames) > 1:
    print("  ⚠ More than one hostname seen — node-pinning did not fully isolate."
          " The experiment is invalid; check why nodelist constraint did not stick.")
else:
    print("  ✓ Single hostname confirmed — variance is cross-seed, not cross-node.")

# --- Per-algorithm summary stats (mean ± SD across seeds) ---
print()
print("=" * 70)
print("Per-algorithm summary across 3 seeds (same node)")
print("=" * 70)
summary_rows = []
for algo, mu in ALGOS:
    sub = df[df["algorithm"] == algo]
    if not len(sub):
        continue
    mean_macro = float(sub["macro_f1"].mean())
    sd_macro   = float(sub["macro_f1"].std(ddof=1)) if len(sub) > 1 else 0.0
    mean_rare  = float(sub["rare_avg_f1"].mean())
    sd_rare    = float(sub["rare_avg_f1"].std(ddof=1)) if len(sub) > 1 else 0.0
    print(f"  {algo:>8}:  macro-F1 = {mean_macro:.4f} ± {sd_macro:.4f}   "
          f"rare-avg-F1 = {mean_rare:.4f} ± {sd_rare:.4f}   "
          f"selected_rounds = {list(sub['selected_round'])}")
    summary_rows.append(dict(algorithm=algo,
                              macro_mean=mean_macro, macro_sd=sd_macro,
                              rare_mean=mean_rare,   rare_sd=sd_rare))

# --- Decision rule ---
print()
print("=" * 70)
print("Decision rule")
print("=" * 70)
if len(summary_rows) == 2:
    a, b = summary_rows
    gap = abs(a["macro_mean"] - b["macro_mean"])
    pooled_sd = np.sqrt(0.5 * (a["macro_sd"] ** 2 + b["macro_sd"] ** 2))
    threshold = 2 * pooled_sd
    print(f"  |Δ_means|  = |{a['macro_mean']:.4f} − {b['macro_mean']:.4f}| = {gap:.4f}")
    print(f"  pooled SD  = {pooled_sd:.4f}")
    print(f"  threshold  = 2 × SD = {threshold:.4f}")
    if gap > threshold:
        print(f"  → |Δ| > 2 × SD: real algorithmic effect on this node (worth Stage B).")
    else:
        print(f"  → |Δ| ≤ 2 × SD: algorithms are statistically indistinguishable here.")
        print(f"    The original single-seed deficit was within seed-level noise.")

# --- Figure: 2-panel ---
COLOR_FA = "#7FBF94"
COLOR_FP = "#3D5A80"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5),
                                gridspec_kw={"wspace": 0.25})

# Panel A: macro-F1 dot plot
xs, ys, cols, labels = [], [], [], []
for i, (algo, _) in enumerate(ALGOS):
    sub = df[df["algorithm"] == algo]
    for _, r in sub.iterrows():
        xs.append(i + np.random.RandomState(int(r["seed"])).uniform(-0.10, 0.10))
        ys.append(r["macro_f1"])
        cols.append(COLOR_FA if algo == "fedavg" else COLOR_FP)
        labels.append(f"s{int(r['seed'])}")
axA.scatter(xs, ys, c=cols, s=90, edgecolor="white", linewidth=1.0, zorder=3)
for i, (algo, _) in enumerate(ALGOS):
    sub = df[df["algorithm"] == algo]
    mean = float(sub["macro_f1"].mean())
    sd   = float(sub["macro_f1"].std(ddof=1)) if len(sub) > 1 else 0.0
    color = COLOR_FA if algo == "fedavg" else COLOR_FP
    axA.errorbar([i], [mean], yerr=[sd], fmt="_", color=color,
                 markersize=30, linewidth=2, capsize=10, zorder=2)
    axA.text(i, mean + sd + 0.005, f"{mean:.3f}", ha="center", va="bottom",
             fontsize=10, color=color, fontweight="bold")
for x, y, lab in zip(xs, ys, labels):
    axA.text(x + 0.04, y, lab, fontsize=7, color="#666", va="center")
axA.set_xticks([0, 1]); axA.set_xticklabels(["FedAvg", "FedProx (μ=0.01)"])
axA.set_ylabel("Test macro-F1")
axA.set_title("(a) L4 macro-F1, 3 seeds on the same physical node",
              loc="left", fontweight="bold", fontsize=11)
# Reference: ±0.04 noise floor observed across HPC nodes on seed 42
axA.axhline(df["macro_f1"].mean(), color="#888", linestyle=":", linewidth=0.8, alpha=0.5)
axA.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)

# Panel B: per-class mean ± SD
classes_to_plot = list(range(7))
x = np.arange(7)
w = 0.36
for i, (algo, _) in enumerate(ALGOS):
    sub = df[df["algorithm"] == algo]
    means = [sub[f"f1_{CLASS_NAMES[c]}"].mean() for c in classes_to_plot]
    sds   = [sub[f"f1_{CLASS_NAMES[c]}"].std(ddof=1) if len(sub) > 1 else 0.0
             for c in classes_to_plot]
    color = COLOR_FA if algo == "fedavg" else COLOR_FP
    label = "FedAvg" if algo == "fedavg" else r"FedProx ($\mu=0.01$)"
    axB.bar(x + (i - 0.5) * w, means, w, yerr=sds, color=color, edgecolor="white",
            linewidth=0.5, capsize=4,
            error_kw=dict(linewidth=1.0, ecolor="#333"),
            label=label)
for c in RARE_IDX:
    axB.axvspan(c - 0.5, c + 0.5, color="#C9A227", alpha=0.10)
axB.set_xticks(x); axB.set_xticklabels(CLASS_NAMES, rotation=20, ha="right", fontsize=8)
axB.set_ylabel("Per-class F1 (mean ± SD, 3 seeds)")
axB.set_title("(b) Per-class F1 on L4, 3-seed mean ± SD  —  rare classes shaded",
              loc="left", fontweight="bold", fontsize=11)
axB.legend(loc="upper left", frameon=False, fontsize=9)
axB.set_ylim(0, 1.0)
axB.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)

host_str = ", ".join(sorted(hostnames)) if hostnames else "unknown"
fig.suptitle(f"Node-pinned 3-seed L4 (node: {host_str})  —  variance isolation",
             fontsize=12, fontweight="bold", y=1.02)
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"F_node_pinned_L4.{ext}")
plt.close(fig)
print()
print(f"Wrote {OUT_FIG / 'F_node_pinned_L4.pdf'}")
print()
print("Done.")
