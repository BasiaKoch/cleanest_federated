"""Phase 3 Experiment P3 — Prospective validation of the collapse detector.

The original Finding 11 (analyse_early_warning_collapse.py) used
within-mix cross-validation, which is LEAKY (the same experiment
families that built the detector also validated it). For
publishability, we need:
  (a) PROSPECTIVE validation on a held-out experiment family
  (b) Comparison to the trivial-baseline (single-feature threshold)
  (c) Operational cost-savings curve

Held-out family: ALL μ-sensitivity runs (mu_sweep_ladder/) — these
are the cleanest "unseen" experimental regime because the μ
sensitivity sweep tests a different question than the rest of the
data (perfect-storm, Li 2020 §5.2, etc.).

Reframing: this is no longer marketed as a "tool" (because the
single-feature baseline matches the multivariate model). It's
marketed as a DESCRIPTIVE observation: "round-20 macro-F1 < 0.30
predicts final per-class collapse with 80% precision."

Output:
  - early_warning_prospective_results.csv
  - F_early_warning_prospective.{pdf,png}  : ROC + cost-savings curve
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[4]
OUT_DIR = REPO_ROOT / "fl_dermamnist/results/thesis_ready/data"
OUT_FIG = REPO_ROOT / "fl_dermamnist/results/thesis_ready/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

# Read the existing early-warning feature table built by the original analyse_early_warning_collapse.py
FEAT_FILE = OUT_DIR / "early_warning_features.csv"
if not FEAT_FILE.exists():
    print(f"FATAL: feature file not found: {FEAT_FILE}")
    print(f"Run analyse_early_warning_collapse.py first.")
    raise SystemExit(1)

df = pd.read_csv(FEAT_FILE)
print(f"Loaded {len(df)} runs with features.")

# Hold out the μ-sweep experiment family for prospective validation.
holdout_mask = df["source"] == "mu_sweep_ladder"
train_df = df[~holdout_mask].copy()
test_df = df[holdout_mask].copy()
print()
print("=" * 80)
print("Prospective validation split")
print("=" * 80)
print(f"  Train (held-IN experiments)  : {len(train_df)} runs, "
      f"{int(train_df['collapsed'].sum())} positives ({train_df['collapsed'].mean()*100:.0f}%)")
print(f"  Test  (held-OUT μ-sweep)     : {len(test_df)} runs, "
      f"{int(test_df['collapsed'].sum())} positives ({test_df['collapsed'].mean()*100:.0f}%)")

feature_cols = ["macro_f1_r20", "min_class_f1_r20", "max_class_f1_r20",
                "range_class_f1_r20", "n_classes_below_r20"] + \
               [f"class{c}_f1_r20" for c in range(7)]

X_tr = train_df[feature_cols].values
y_tr = train_df["collapsed"].astype(int).values
X_te = test_df[feature_cols].values
y_te = test_df["collapsed"].astype(int).values

# Skip if test fold has only one class
if int(y_te.sum()) == 0 or int(y_te.sum()) == len(y_te):
    print()
    print(f"WARNING: held-out fold has only {int(y_te.sum())} positives out of {len(y_te)} — single-class fold")
    print(f"Reporting train-only AUC.")
    clf = LogisticRegression(max_iter=10000, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    auc_tr = roc_auc_score(y_tr, clf.predict_proba(X_tr)[:, 1])
    print(f"  Train AUC: {auc_tr:.3f}")
    print(f"  (prospective ROC cannot be computed on single-class fold)")
    auc_test = float("nan")
    test_probs = clf.predict_proba(X_te)[:, 1]
else:
    # Train multivariate logistic regression
    clf = LogisticRegression(max_iter=10000, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    train_probs = clf.predict_proba(X_tr)[:, 1]
    test_probs = clf.predict_proba(X_te)[:, 1]
    auc_train = roc_auc_score(y_tr, train_probs)
    auc_test = roc_auc_score(y_te, test_probs)
    print(f"  Multivariate logreg:  train AUC = {auc_train:.3f}, test AUC = {auc_test:.3f}")

# ----------------------------------------------------------------
# Single-feature baseline: just macro_f1_r20 (the simplest threshold)
# ----------------------------------------------------------------
if int(y_te.sum()) > 0 and int(y_te.sum()) < len(y_te):
    base_train_auc = roc_auc_score(y_tr, -train_df["macro_f1_r20"].values)
    base_test_auc = roc_auc_score(y_te, -test_df["macro_f1_r20"].values)
    print(f"  Simple threshold (macro_f1_r20 < τ): train AUC = {base_train_auc:.3f}, test AUC = {base_test_auc:.3f}")

# ----------------------------------------------------------------
# Operating-point analysis on the TEST set
# ----------------------------------------------------------------
print()
print("=" * 80)
print("Operating-point analysis on prospective test set")
print("=" * 80)
ROUND_BUDGET = 150
SAVED_ROUNDS = ROUND_BUDGET - 20  # rounds saved by early-stopping at round 20
op_rows = []
for thr in [0.3, 0.5, 0.7, 0.85]:
    preds = (test_probs >= thr).astype(int)
    if preds.sum() == 0 or int(y_te.sum()) == 0:
        prec = float("nan"); rec = float("nan")
    else:
        prec = precision_score(y_te, preds, zero_division=0)
        rec = recall_score(y_te, preds, zero_division=0)
    # Compute saved would be: SAVED_ROUNDS × precision (correctly-stopped collapses)
    # minus SAVED_ROUNDS × false_pos_rate × (number of true-non-collapses early-stopped wrongly)
    n_flagged = int(preds.sum())
    n_pos_truly_caught = int(((preds == 1) & (y_te == 1)).sum())
    n_neg_falsely_flagged = int(((preds == 1) & (y_te == 0)).sum())
    net_rounds_saved = n_pos_truly_caught * SAVED_ROUNDS - n_neg_falsely_flagged * SAVED_ROUNDS
    op_rows.append(dict(
        threshold=thr,
        precision=prec, recall=rec,
        n_flagged=n_flagged,
        n_truly_collapsed=n_pos_truly_caught,
        n_falsely_flagged=n_neg_falsely_flagged,
        net_rounds_saved=net_rounds_saved,
    ))
op_df = pd.DataFrame(op_rows)
op_df.to_csv(OUT_DIR / "early_warning_operating_points.csv", index=False)
print(op_df.to_string(index=False))

# ----------------------------------------------------------------
# Save final results
# ----------------------------------------------------------------
summary = dict(
    n_train=len(train_df),
    n_test=len(test_df),
    n_train_positive=int(y_tr.sum()),
    n_test_positive=int(y_te.sum()),
    auc_train=float(roc_auc_score(y_tr, clf.predict_proba(X_tr)[:, 1])) if len(set(y_tr)) > 1 else float("nan"),
    auc_test=float(auc_test),
    auc_test_simple_threshold=float(base_test_auc) if int(y_te.sum()) > 0 and int(y_te.sum()) < len(y_te) else float("nan"),
)
pd.DataFrame([summary]).to_csv(OUT_DIR / "early_warning_prospective_summary.csv", index=False)
print()
print(f"Wrote {OUT_DIR/'early_warning_prospective_summary.csv'}")
print(f"  prospective AUC = {auc_test:.3f}")

# ----------------------------------------------------------------
# Figure: ROC on test set + cost-savings curve
# ----------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"wspace": 0.25})

# Panel A: prospective ROC on held-out μ-sweep
if int(y_te.sum()) > 0 and int(y_te.sum()) < len(y_te):
    fpr, tpr, _ = roc_curve(y_te, test_probs)
    axA.plot(fpr, tpr, color="#3D5A80", linewidth=2.2,
             label=f"Multivariate logreg\n(prospective AUC = {auc_test:.3f})")
    # Simple-feature baseline
    fpr_b, tpr_b, _ = roc_curve(y_te, -test_df["macro_f1_r20"].values)
    axA.plot(fpr_b, tpr_b, color="#C03A2B", linewidth=2.0, linestyle="--",
             label=f"Simple threshold (macro_f1_r20)\n(prospective AUC = {base_test_auc:.3f})")
    axA.plot([0, 1], [0, 1], color="#888", linewidth=1, linestyle=":",
             label="Random (AUC = 0.5)")
    axA.set_xlabel("False positive rate")
    axA.set_ylabel("True positive rate")
    axA.set_title(f"(a) Prospective ROC on held-out μ-sweep ({len(test_df)} runs)",
                  loc="left", fontweight="bold", fontsize=11)
    axA.legend(loc="lower right", frameon=False, fontsize=9)
    axA.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)

# Panel B: operating-point cost-savings curve
op_df_plot = op_df.dropna(subset=["precision", "recall"])
if len(op_df_plot):
    axB.bar(np.arange(len(op_df_plot)), op_df_plot["net_rounds_saved"],
            color=["#1f6f3f" if v > 0 else "#b04040" for v in op_df_plot["net_rounds_saved"]],
            edgecolor="white", linewidth=0.6)
    for i, row in enumerate(op_df_plot.itertuples()):
        axB.text(i, row.net_rounds_saved + 5,
                 f"P={row.precision:.2f}\nR={row.recall:.2f}",
                 ha="center", va="bottom", fontsize=8.5)
    axB.set_xticks(np.arange(len(op_df_plot)))
    axB.set_xticklabels([f"thr={t:.2f}" for t in op_df_plot["threshold"]], fontsize=9)
    axB.axhline(0, color="#444", linewidth=0.8)
    axB.set_ylabel("Net rounds saved (positive = compute saved)")
    axB.set_title("(b) Net compute saved by early-stopping flagged runs",
                  loc="left", fontweight="bold", fontsize=11)
    axB.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
    axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)

fig.suptitle("Prospective validation of round-20 collapse-prediction detector "
             "(Phase 3 P3, held-out μ-sweep family)",
             fontsize=11.5, fontweight="bold", y=1.02)
for ext in ("pdf", "png"):
    fig.savefig(OUT_FIG / f"F_early_warning_prospective.{ext}")
plt.close(fig)
print()
print(f"Wrote {OUT_FIG/'F_early_warning_prospective.pdf'}")
print()
print("Done.")
