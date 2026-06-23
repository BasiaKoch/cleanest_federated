"""Confusion-matrix analysis across L4 protocols.

Loads 91 test_predictions_*.npz files across L4 experiments and
produces clinically-interpretable confusion-matrix views answering
two specific questions:

  Q1 (D1 LR-asymmetry): under 5:1 LR asymmetry, where do FedAvg /
     FedProx send rare-class test inputs that they fail to predict?
     Does FedNova send them correctly?

  Q2 (Li 2020 §5.2): under the asymmetric-straggler protocol, how
     does FedAvg+drop reroute rare-class predictions, and how does
     FedProx+γ-inexact rescue them?

Test set class distribution:
   actinic=66, basal=103, benign-kerat=220, dermato=23 (rare),
   melanoma=223 (rare), mel-nevi=1341 (dominant 67%),
   vascular=29 (rare). Total = 2005.

Output:
   F_l4_confusion_d1_5_1.{pdf,png}        Rare-class fate under D1 5:1
   F_l4_confusion_li2020.{pdf,png}        4-condition mechanism
   F_l4_diagonal_heatmap.{pdf,png}        Per-class accuracy heatmap
                                          across all protocols
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[4]
RESULTS = REPO_ROOT / "fl_dermamnist/results"
OUT_DIR = REPO_ROOT / "fl_dermamnist/results/thesis_ready/data"
OUT_FIG = REPO_ROOT / "fl_dermamnist/results/thesis_ready/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["actinic", "basal", "b-kerat", "dermato", "melanoma", "mel-nevi", "vascular"]
RARE_IDX = (3, 4, 6)
COMMON_IDX = (0, 1, 2, 5)
NUM_CLASSES = 7


def _load_predictions(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path)
    return d["targets"].astype(int), d["predictions"].astype(int)


def _confusion_row_normalised(npz_path: Path) -> np.ndarray:
    """Compute row-normalised confusion matrix (each row sums to 1)."""
    targets, preds = _load_predictions(npz_path)
    cm = confusion_matrix(targets, preds, labels=list(range(NUM_CLASSES)))
    row_sums = cm.sum(axis=1, keepdims=True)
    return cm / np.where(row_sums == 0, 1, row_sums)


def _avg_confusion_across_seeds(npz_paths: list[Path]) -> np.ndarray:
    """Average row-normalised confusion matrices across seeds."""
    if not npz_paths:
        return None
    mats = [_confusion_row_normalised(p) for p in npz_paths if p.exists()]
    if not mats:
        return None
    return np.mean(mats, axis=0)


# ----------------------------------------------------------------
# Figure A: D1 LR asymmetry — where do rare-class inputs go?
# Layout: 3 cols (FedAvg, FedProx, FedNova) × 2 rows (1:1 symmetric, 5:1 asymmetric)
# ----------------------------------------------------------------
print("=" * 80)
print("Figure A: D1 LR asymmetry confusion matrices")
print("=" * 80)

D1_DIR = RESULTS / "asymmetric_lr_L4"
SEEDS = [42, 123, 456]


def _d1_files(algo: str, mu_str: str, lr_pc: str | None) -> list[Path]:
    """Locate D1 prediction files matching the spec."""
    files = []
    for seed in SEEDS:
        if lr_pc is None:
            # Symmetric — no lr-per-client tag
            f = D1_DIR / f"test_predictions_{algo}_mu{mu_str}_E20_s{seed}.npz"
        else:
            f = D1_DIR / f"test_predictions_{algo}_mu{mu_str}_E20_lrPC-{lr_pc}_s{seed}.npz"
        if f.exists():
            files.append(f)
    return files


fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)

ALGOS = [("fedavg", "0.0"), ("fedprox", "0.01"), ("fednova", "0.0")]
ALGO_LABELS = ["FedAvg", r"FedProx ($\mu=0.01$)", "FedNova"]
RATIO_LABELS = ["1:1 symmetric", "5:1 (Client 1 LR=0.002)"]
RATIO_TAGS = [None, "c0lr0.01-c1lr0.002"]

for col, ((algo, mu_str), algo_label) in enumerate(zip(ALGOS, ALGO_LABELS)):
    for row, (ratio_label, lr_pc) in enumerate(zip(RATIO_LABELS, RATIO_TAGS)):
        ax = axes[row, col]
        files = _d1_files(algo, mu_str, lr_pc)
        cm = _avg_confusion_across_seeds(files)
        if cm is None:
            ax.text(0.5, 0.5, f"No data\n({algo}, {ratio_label})",
                    ha="center", va="center", transform=ax.transAxes)
            continue
        # Plot
        im = ax.imshow(cm, cmap="Greens", vmin=0, vmax=1, aspect="auto")
        # Annotate cells with values
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                v = cm[i, j]
                if v >= 0.01:
                    color = "white" if v >= 0.5 else "#333"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7.5, color=color, fontweight="bold" if i == j else "normal")
        # Highlight rare classes
        for ri in RARE_IDX:
            ax.add_patch(plt.Rectangle((-0.5, ri - 0.5), NUM_CLASSES, 1,
                                       fill=False, edgecolor="#C9A227", linewidth=1.5))
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_yticklabels(CLASS_NAMES, fontsize=8)
        ax.set_title(f"{algo_label}  —  {ratio_label}",
                     loc="left", fontweight="bold", fontsize=10)
        if col == 0:
            ax.set_ylabel("True class", fontsize=9)
        if row == 1:
            ax.set_xlabel("Predicted class", fontsize=9)

fig.suptitle("D1: where rare-class inputs are sent under symmetric vs 5:1 LR asymmetry  —  "
             "yellow rows = rare classes (dermato, melanoma, vascular)",
             fontsize=11.5, fontweight="bold", y=1.01)
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"F_l4_confusion_d1_5_1.{ext}")
plt.close(fig)
print(f"Wrote {OUT_FIG/'F_l4_confusion_d1_5_1.pdf'}")

# ----------------------------------------------------------------
# Figure B: Four-condition L4 — 4 conditions side by side
# ----------------------------------------------------------------
print()
print("=" * 80)
print("Figure B: Four-condition L4 protocol confusion matrices")
print("=" * 80)

LI_DIR = RESULTS / "li2020_asymmetric_L4"
LI_CONDITIONS = [
    ("FedAvg, no straggler", "fedavg",  "0.0",  ""),
    ("FedAvg + drop", "fedavg",  "0.0",  "_sh-fixed_stragglers_drop"),
    ("FedProx + γ-inexact", "fedprox", "0.01", "_sh-fixed_stragglers"),
    ("FedProx + drop (control)", "fedprox", "0.01", "_sh-fixed_stragglers_drop"),
]


def _li_files(algo: str, mu_str: str, tag: str) -> list[Path]:
    files = []
    for seed in SEEDS:
        f = LI_DIR / f"test_predictions_{algo}_mu{mu_str}_E20{tag}_s{seed}.npz"
        if f.exists():
            files.append(f)
    return files


fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), sharey=True)

for col, (label, algo, mu_str, tag) in enumerate(LI_CONDITIONS):
    ax = axes[col]
    files = _li_files(algo, mu_str, tag)
    cm = _avg_confusion_across_seeds(files)
    if cm is None:
        ax.text(0.5, 0.5, f"No data", ha="center", va="center", transform=ax.transAxes)
        continue
    im = ax.imshow(cm, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = cm[i, j]
            if v >= 0.01:
                color = "white" if v >= 0.5 else "#333"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold" if i == j else "normal")
    for ri in RARE_IDX:
        ax.add_patch(plt.Rectangle((-0.5, ri - 0.5), NUM_CLASSES, 1,
                                   fill=False, edgecolor="#C9A227", linewidth=1.5))
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontsize=7.5)
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(CLASS_NAMES, fontsize=7.5)
    ax.set_title(label, loc="left", fontweight="bold", fontsize=9)
    if col == 0:
        ax.set_ylabel("True class", fontsize=9)
    ax.set_xlabel("Predicted class", fontsize=9)

fig.suptitle("Four-condition L4: FedAvg+drop reroutes rare-class predictions into mel-nevi, "
             "while FedProx+γ-inexact rescues them (rare classes outlined in yellow)",
             fontsize=11.5, fontweight="bold", y=1.04)
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"F_l4_confusion_li2020.{ext}")
plt.close(fig)
print(f"Wrote {OUT_FIG/'F_l4_confusion_li2020.pdf'}")

# ----------------------------------------------------------------
# Figure C: per-class accuracy heatmap (the diagonal across all protocols)
# Aggregate ALL L4 prediction files and report per-(experiment × algorithm) per-class accuracy
# ----------------------------------------------------------------
print()
print("=" * 80)
print("Figure C: Per-class accuracy heatmap across protocols")
print("=" * 80)

# Walk all L4 dirs and compute per-(experiment, algorithm, condition) per-class accuracy
EXPERIMENTS = {
    "Heterogeneity ladder": RESULTS / "heterogeneity_ladder/L4_two_client_90_10_rare_stress",
    "Node-pinned symmetric": RESULTS / "node_pinned_L4",
    "Engineered 90/10 baseline": RESULTS / "two_client_90_10_rare_stress",
    "Li 2020 §5.2 (FA+drop)": LI_DIR,
    "Li 2020 §5.2 (FP+γ-inexact)": LI_DIR,
    "Perfect-storm (FA+drop)": RESULTS / "fedprox_perfect_storm_L4",
    "Perfect-storm (FP μ=1.0)": RESULTS / "fedprox_perfect_storm_L4",
    "Perfect-storm (FP μ=0.01)": RESULTS / "fedprox_perfect_storm_L4",
    "D1 1:1 FedAvg": D1_DIR,
    "D1 5:1 FedAvg": D1_DIR,
    "D1 1:1 FedProx": D1_DIR,
    "D1 5:1 FedProx": D1_DIR,
    "D1 1:1 FedNova": D1_DIR,
    "D1 5:1 FedNova": D1_DIR,
}
FILE_PATTERNS = {
    "Heterogeneity ladder": ["test_predictions_fedavg_mu0.0_E20_s42.npz"],
    "Node-pinned symmetric": [f"test_predictions_fedavg_mu0.0_E20_s{s}.npz" for s in SEEDS],
    "Engineered 90/10 baseline": [f"test_predictions_fedavg_mu0.0_E20_s{s}.npz" for s in SEEDS],
    "Li 2020 §5.2 (FA+drop)": [f"test_predictions_fedavg_mu0.0_E20_sh-fixed_stragglers_drop_s{s}.npz" for s in SEEDS],
    "Li 2020 §5.2 (FP+γ-inexact)": [f"test_predictions_fedprox_mu0.01_E20_sh-fixed_stragglers_s{s}.npz" for s in SEEDS],
    "Perfect-storm (FA+drop)": [f"test_predictions_fedavg_mu0.0_E20_sh-random_stragglers_drop_s{s}.npz" for s in SEEDS],
    "Perfect-storm (FP μ=1.0)": [f"test_predictions_fedprox_mu1.0_E20_sh-random_stragglers_s{s}.npz" for s in SEEDS],
    "Perfect-storm (FP μ=0.01)": [f"test_predictions_fedprox_mu0.01_E20_sh-random_stragglers_s{s}.npz" for s in SEEDS],
    "D1 1:1 FedAvg": [f"test_predictions_fedavg_mu0.0_E20_s{s}.npz" for s in SEEDS],
    "D1 5:1 FedAvg": [f"test_predictions_fedavg_mu0.0_E20_lrPC-c0lr0.01-c1lr0.002_s{s}.npz" for s in SEEDS],
    "D1 1:1 FedProx": [f"test_predictions_fedprox_mu0.01_E20_s{s}.npz" for s in SEEDS],
    "D1 5:1 FedProx": [f"test_predictions_fedprox_mu0.01_E20_lrPC-c0lr0.01-c1lr0.002_s{s}.npz" for s in SEEDS],
    "D1 1:1 FedNova": [f"test_predictions_fednova_mu0.0_E20_s{s}.npz" for s in SEEDS],
    "D1 5:1 FedNova": [f"test_predictions_fednova_mu0.0_E20_lrPC-c0lr0.01-c1lr0.002_s{s}.npz" for s in SEEDS],
}

rows = []
for label, root in EXPERIMENTS.items():
    files = [root / p for p in FILE_PATTERNS[label]]
    cm = _avg_confusion_across_seeds(files)
    if cm is None:
        continue
    diag = np.diag(cm)
    rows.append(dict(experiment=label, **{n: float(diag[i]) for i, n in enumerate(CLASS_NAMES)}))

diag_df = pd.DataFrame(rows)
diag_df.to_csv(OUT_DIR / "l4_per_class_accuracy.csv", index=False)
print(diag_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# Heatmap
fig, ax = plt.subplots(1, 1, figsize=(11, 8))
mat = diag_df[CLASS_NAMES].values
exp_labels = diag_df["experiment"].tolist()
im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
for i in range(len(exp_labels)):
    for j in range(NUM_CLASSES):
        v = mat[i, j]
        color = "white" if v < 0.3 else "black"
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=color)
# Mark rare-class columns
for ci in RARE_IDX:
    ax.axvline(ci - 0.5, color="#C9A227", linewidth=1.5, alpha=0.6)
    ax.axvline(ci + 0.5, color="#C9A227", linewidth=1.5, alpha=0.6)
ax.set_xticks(range(NUM_CLASSES))
ax.set_xticklabels(CLASS_NAMES, rotation=20, ha="right", fontsize=9)
ax.set_yticks(range(len(exp_labels)))
ax.set_yticklabels(exp_labels, fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Per-class accuracy (diagonal of confusion matrix)")
ax.set_title("Per-class test accuracy across L4 protocols  —  yellow lines bracket rare classes",
             loc="left", fontweight="bold", fontsize=11)
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"F_l4_diagonal_heatmap.{ext}")
plt.close(fig)
print()
print(f"Wrote {OUT_FIG/'F_l4_diagonal_heatmap.pdf'}")

# ----------------------------------------------------------------
# Print key insights extracted from the confusion matrices
# ----------------------------------------------------------------
print()
print("=" * 80)
print("KEY CONFUSION-MATRIX INSIGHTS (rare classes only)")
print("=" * 80)
# For each protocol, what fraction of true-rare-class inputs are predicted as nevi (the dominant class)?
print()
print("Fraction of rare-class inputs misrouted to mel-nevi (the dominant 67% class):")
for label, root in EXPERIMENTS.items():
    files = [root / p for p in FILE_PATTERNS[label]]
    cm = _avg_confusion_across_seeds(files)
    if cm is None:
        continue
    nevi_pred_for_rare = cm[list(RARE_IDX), 5].mean()  # col 5 = nevi
    correct_for_rare = cm[list(RARE_IDX), list(RARE_IDX)].mean()
    print(f"  {label:<30s}: misrouted to nevi = {nevi_pred_for_rare:.2f}, correct = {correct_for_rare:.2f}")

print()
print("Done.")
